"""
validate.py — Pre-solve network validation (mirrors the Excel VALIDATION sheet).
"""
import math
from typing import List, Tuple, Dict
from network import Node, Pipe, check_connectivity, straight_line_distance


def validate_network(nodes: List[Node], pipes: List[Pipe],
                     verbose: bool = True) -> Tuple[bool, List[str]]:
    errors, warnings = [], []
    node_map = {n.id: n for n in nodes}

    def err(msg):  errors.append(f"[FAIL] {msg}")
    def warn(msg): warnings.append(f"[!]   {msg}")

    # Guard: empty inputs
    if not nodes:
        err("NODES sheet is empty — no nodes were read")
    if not pipes:
        err("PIPES sheet is empty — no pipes were read")
    if not nodes or not pipes:
        if verbose:
            print(f"\n{'='*60}\n  PRE-SOLVE VALIDATION\n{'='*60}")
            for e in errors: print(f"  {e}")
            print(f"{'='*60}\n")
        return False, errors

    sources  = [n for n in nodes if n.is_source()]
    offtakes = [n for n in nodes if n.is_offtake()]

    # V01 At least one source
    if not sources:
        err("No source nodes — add at least one PRS/PRV as source")

    # V02 At least one offtake
    if not offtakes:
        err("No offtake nodes — add at least one demand node")

    # V03 Sources have known pressure
    for n in sources:
        if n.pressure <= 101_325.0:
            err(f"Source {n.id} ({n.label}): pressure <= atmospheric — "
                f"Known_P_kPa_g must be a positive number (e.g. 400). "
                f"Check the cell contains a plain value, not a formula or text.")

    # V04 Sources have no demand
    for n in sources:
        if abs(n.demand) > 1e-12:
            err(f"Source {n.id} ({n.label}) has demand "
                f"{n.demand*3600:.1f} Sm3/h -- sources supply, not consume")

    # V05 Offtakes have positive demand
    for n in offtakes:
        if n.demand <= 0:
            err(f"Offtake {n.id} ({n.label}) has zero/negative demand")

    # V06 All pipe nodes exist
    for pipe in pipes:
        if pipe.from_node not in node_map:
            err(f"Pipe {pipe.id}: From_Node '{pipe.from_node}' not in NODES")
        if pipe.to_node not in node_map:
            err(f"Pipe {pipe.id}: To_Node '{pipe.to_node}' not in NODES")

    # V07 No self-loops
    for pipe in pipes:
        if pipe.from_node == pipe.to_node:
            err(f"Pipe {pipe.id}: From_Node == To_Node (self-loop)")

    # V08 No duplicate pipe pairs  ← must contain word "duplicate"
    seen: Dict[tuple, str] = {}
    for pipe in pipes:
        key = tuple(sorted([pipe.from_node, pipe.to_node]))
        if key in seen:
            err(f"Duplicate pipe pair: {seen[key]} and {pipe.id} both connect "
                f"{pipe.from_node} <-> {pipe.to_node} -- delete one")
        else:
            seen[key] = pipe.id

    # V09 Positive dimensions
    for pipe in pipes:
        if pipe.length   <= 0: err(f"Pipe {pipe.id}: length <= 0")
        if pipe.diameter <= 0: err(f"Pipe {pipe.id}: diameter <= 0")
        if pipe.roughness < 0: err(f"Pipe {pipe.id}: roughness < 0")

    # V10 Length ≥ straight-line distance
    for pipe in pipes:
        if pipe.from_node not in node_map or pipe.to_node not in node_map:
            continue
        nf, nt = node_map[pipe.from_node], node_map[pipe.to_node]
        if nf.x_coord == 0 and nf.y_coord == 0:
            continue
        sl = straight_line_distance(nf, nt)
        if sl > 1.0:
            if pipe.length < sl - 0.5:
                err(f"Pipe {pipe.id}: length {pipe.length:.0f} m < "
                    f"straight-line {sl:.0f} m -- physically impossible")
            elif pipe.length / sl > 2.5:
                warn(f"Pipe {pipe.id}: route factor "
                     f"{pipe.length/sl:.1f}x -- verify as-built")

    # V11 Collinearity check
    for pipe in pipes:
        if pipe.from_node not in node_map or pipe.to_node not in node_map:
            continue
        nf, nt = node_map[pipe.from_node], node_map[pipe.to_node]
        dx, dy = nt.x_coord-nf.x_coord, nt.y_coord-nf.y_coord
        for nid, n in node_map.items():
            if nid in [pipe.from_node, pipe.to_node]: continue
            cross = dx*(n.y_coord-nf.y_coord) - dy*(n.x_coord-nf.x_coord)
            if abs(cross) < 1.0 and (dx != 0 or dy != 0):
                denom = dx if dx != 0 else dy
                num   = (n.x_coord-nf.x_coord) if dx != 0 else (n.y_coord-nf.y_coord)
                t = num/denom if denom else 0.5
                if 0.01 < t < 0.99:
                    err(f"Node {n.id} ({n.label}) lies ON pipe {pipe.id} "
                        f"({nf.label}->{nt.label}) -- split the pipe at {n.label}")

    # V12 Connectivity (skip if pipe nodes are broken — check_connectivity is safe now)
    conn = check_connectivity(nodes, pipes)
    if not conn['connected']:
        for nid in conn['isolated']:
            n = node_map[nid]
            err(f"Node {nid} ({n.label}) not reachable from any source")

    # V13 Total demand > 0
    tot_demand = sum(n.demand for n in offtakes)
    if tot_demand <= 0:
        err("Total network demand is zero")

    # V14 Min pressure feasibility
    if sources:
        max_src_p = max(n.pressure for n in sources)
        for n in offtakes:
            if n.min_pressure > max_src_p:
                warn(f"Node {n.id} min pressure "
                     f"({(n.min_pressure-101325)/1e3:.1f} kPa(g)) "
                     f"exceeds max source pressure "
                     f"({(max_src_p-101325)/1e3:.1f} kPa(g))")

    # V15 GIS coordinates present (needed for geometric checks)
    all_zero = all(n.x_coord == 0.0 and n.y_coord == 0.0 for n in nodes)
    if all_zero:
        warn("All node X_Coord/Y_Coord are zero — pipe length geometry checks skipped. "
             "Add coordinates to enable V10 (length vs straight-line) validation.")

    passed = len(errors) == 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  PRE-SOLVE VALIDATION")
        print(f"{'='*60}")
        if passed:
            print(f"  [OK] ALL CHECKS PASSED -- safe to run solver")
        else:
            print(f"  [FAIL] {len(errors)} error(s) -- fix before running solver")
        if warnings:
            print(f"  [!] {len(warnings)} warning(s)")
        for e in errors:   print(f"  {e}")
        for w in warnings: print(f"  {w}")
        tot_h = tot_demand * 3600 if offtakes else 0
        print(f"\n  Network: {len(nodes)} nodes | {len(pipes)} pipes")
        print(f"  Total effective demand: {tot_h:.1f} Sm3/h")
        print(f"{'='*60}\n")

    return passed, errors + warnings
