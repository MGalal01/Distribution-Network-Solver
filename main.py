"""
main.py
-------
Entry point for the PE80 gas distribution network hydraulic solver.

Usage
-----
    python main.py --input PE80_Network_Template.xlsx
    python main.py --input network.xlsx --output results.xlsx --verbose
    python main.py --input network.xlsx --sg 0.62 --temp 20 --pressure 400

Arguments
---------
    --input     Path to Excel input file (required)
    --output    Path to write results  (default: same file, adds RESULTS sheet)
    --sg        Gas specific gravity   (default: 0.62)
    --temp      Flowing temperature    [°C]  (default: 20)
    --pressure  Average network pressure [kPa(g)] for gas property calc (default: 300)
    --tol       Solver tolerance [Sm³/s] (default: 1e-6)
    --maxiter   Maximum iterations     (default: 100)
    --map       Generate HTML pressure map (default: False)
    --verbose   Print convergence table (default: True)
"""

import argparse
import sys
import os
import pathlib
import logging
import shutil

# ── Windows-safe UTF-8 output ────────────────────────────────────────────────
# Prevents UnicodeEncodeError on Windows cmd/PowerShell (cp1252 default).
# Tries three escalating strategies so it works on Python 3.7+ everywhere.
import io as _io
def _ensure_utf8(stream_name: str) -> None:
    stream = getattr(sys, stream_name)
    # Strategy 1: reconfigure()  (Python 3.7+, most environments)
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
            return
        except Exception:
            pass
    # Strategy 2: wrap the underlying buffer
    if hasattr(stream, 'buffer'):
        try:
            wrapped = _io.TextIOWrapper(
                stream.buffer, encoding='utf-8',
                errors='replace', line_buffering=True)
            setattr(sys, stream_name, wrapped)
            return
        except Exception:
            pass
    # Strategy 3: give up gracefully — don't crash, just lose fancy chars

for _s in ('stdout', 'stderr'):
    _ensure_utf8(_s)
del _ensure_utf8, _s

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))

from gas_properties import compute_gas_properties, P_STD
from io_excel        import read_nodes, read_pipes, write_results
from validate        import validate_network
from solver          import solve_network
from postprocess     import print_summary, generate_pressure_map, generate_flow_map


logging.basicConfig(
    level=logging.INFO,
    format='  [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="PE80 Gas Distribution Network — Newton-Raphson Hydraulic Solver"
    )
    parser.add_argument('--input',    required=True,
                        help='Path to Excel input file (PE80_Network_Template.xlsx)')
    parser.add_argument('--output',   default=None,
                        help='Output file path (default: input file with RESULTS sheet added)')
    parser.add_argument('--sg',       type=float, default=0.62,
                        help='Gas specific gravity relative to air (default: 0.62)')
    parser.add_argument('--temp',     type=float, default=20.0,
                        help='Flowing temperature [°C] (default: 20)')
    parser.add_argument('--pressure', type=float, default=300.0,
                        help='Representative network pressure for gas props [kPa(g)] (default: 300)')
    parser.add_argument('--tol',      type=float, default=1e-6,
                        help='Convergence tolerance [Sm³/s] (default: 1e-6)')
    parser.add_argument('--maxiter',  type=int,   default=100,
                        help='Maximum Newton-Raphson iterations (default: 100)')
    parser.add_argument('--map',      action='store_true', default=True,
                        help='Generate interactive HTML pressure map (default: True)')
    parser.add_argument('--verbose',  action='store_true', default=True,
                        help='Print convergence table (default: True)')
    # If launched with no arguments (e.g. from IDE), auto-detect a single Excel
    # file in the script directory rather than crashing with SystemExit: 2.
    if len(sys.argv) == 1:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _xlsx = [f for f in os.listdir(_script_dir)
                 if f.lower().endswith('.xlsx') and 'result' not in f.lower()]
        if len(_xlsx) == 1:
            sys.argv.extend(['--input', os.path.join(_script_dir, _xlsx[0])])
        else:
            parser.print_help()
            sys.exit(0)

    args = parser.parse_args()

    # ── Input file ────────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        log.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Check required sheets exist before doing anything else
    from openpyxl import load_workbook as _lw
    try:
        _wb = _lw(args.input, read_only=True)
        _found = _wb.sheetnames
        _wb.close()
    except Exception as exc:
        log.error(f"Cannot open '{args.input}': {exc}")
        sys.exit(1)
    for _s in ('NODES', 'PIPES'):
        if _s not in _found:
            log.error(f"Sheet '{_s}' not found in '{args.input}'. Sheets present: {_found}")
            sys.exit(1)

    # Check file is writable (not locked by Excel)
    try:
        with open(args.input, 'a'):
            pass
    except PermissionError:
        log.error(f"Cannot write to '{args.input}' -- close the file in Excel first.")
        sys.exit(1)

    output_path = args.output or args.input

    # Use pathlib to build working path — handles any extension correctly
    _inp = pathlib.Path(args.input)
    working_path = str(_inp.with_name(_inp.stem + '_solving' + _inp.suffix)) \
        if output_path == args.input else args.input
    if output_path == args.input:
        shutil.copy(args.input, working_path)

    print(f"\n{'='*60}")
    print(f"  PE80 Network Hydraulic Solver  v1.0")
    print(f"{'='*60}")
    print(f"  Input  : {args.input}")
    print(f"  Output : {output_path}")

    # ── Read network ──────────────────────────────────────────────────────────
    log.info("Reading network data from Excel...")
    nodes = read_nodes(args.input)
    pipes = read_pipes(args.input)

    if not nodes or not pipes:
        log.error("No nodes or pipes found. Check sheet names: NODES, PIPES")
        sys.exit(1)

    # ── Gas properties at representative conditions ───────────────────────────
    T_K     = args.temp + 273.15                        # °C → K
    P_abs   = args.pressure * 1000.0 + P_STD            # kPa(g) → Pa(a)
    gas     = compute_gas_properties(P_abs, T_K, args.sg)

    print(f"\n  Gas properties at {args.temp:.0f}°C, {args.pressure:.0f} kPa(g):")
    print(f"    SG    = {gas.SG:.3f}")
    print(f"    Z     = {gas.Z:.4f}")
    print(f"    rho   = {gas.rho:.3f} kg/m3")
    print(f"    mu    = {gas.mu:.2e} Pa.s")

    # ── Pre-solve validation ──────────────────────────────────────────────────
    passed, issues = validate_network(nodes, pipes, verbose=True)
    if not passed:
        log.error("Validation failed. Fix errors above before running solver.")
        sys.exit(1)

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = solve_network(
        nodes    = nodes,
        pipes    = pipes,
        gas      = gas,
        tol      = args.tol,
        max_iter = args.maxiter,
        verbose  = args.verbose,
    )

    # ── Post-process ──────────────────────────────────────────────────────────
    print_summary(nodes, pipes, result, gas)

    # ── Write results to Excel ────────────────────────────────────────────────
    try:
        write_results(working_path, nodes, pipes, result, gas)
        if working_path != output_path:
            shutil.move(working_path, output_path)
    except Exception:
        if working_path != output_path and os.path.exists(working_path):
            os.remove(working_path)
        raise

    log.info(f"Results written to: {output_path}")

    # ── Optional HTML maps ────────────────────────────────────────────────────
    if args.map:
        _base     = str(pathlib.Path(output_path).with_suffix(''))
        map_path  = _base + '_pressure_map.html'
        flow_path = _base + '_flow_map.html'
        generate_pressure_map(nodes, pipes, map_path)
        generate_flow_map(nodes, pipes, result, gas, flow_path)

    if not result.converged:
        log.warning("Solver did not converge -- results may be unreliable.")
        sys.exit(2)

    print(f"  Done.  {'[OK] Converged' if result.converged else '[FAIL] Did not converge'}\n")


if __name__ == '__main__':
    main()
