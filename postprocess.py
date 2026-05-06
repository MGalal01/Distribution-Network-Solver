"""
postprocess.py
--------------
Compute all engineering outputs from converged solver results.

Per-pipe:  flow, velocity, Re, f, ΔP, pressure gradient, loading ratio
Per-node:  pressure, margin above minimum, supply zone
Network:   mass balance, maximum velocity, minimum pressure margin
"""

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from network        import Node, Pipe
from hydraulics     import compute_reynolds, erosional_velocity, swamee_jain
from gas_properties import GasProperties, sm3s_to_m3s, P_STD


# ── Limits ────────────────────────────────────────────────────────────────────
MAX_GRAD_MBAR_M   = 1.0    # mbar/m  — recommended max gradient for distribution
WARN_LOADING_PCT  = 70.0   # %       — flag pipes above this loading
CRIT_LOADING_PCT  = 90.0   # %       — critical flag


def compute_pipe_results(pipes:  List[Pipe],
                         nodes:  List[Node],
                         result,            # SolverResult
                         gas:    GasProperties) -> pd.DataFrame:
    """
    Compute all per-pipe engineering outputs.

    Parameters
    ----------
    pipes  : list of Pipe   (with .flow populated by solver)
    nodes  : list of Node   (with .pressure populated by solver)
    result : SolverResult
    gas    : GasProperties

    Returns
    -------
    pd.DataFrame  one row per pipe
    """
    node_map = {n.id: n for n in nodes}
    rows = []

    for j, pipe in enumerate(pipes):
        Q_sm3s = pipe.flow                      # Sm³/s
        Q_sm3h = Q_sm3s * 3600.0               # Sm³/h
        Q_abs  = abs(Q_sm3s)

        # Flow direction
        direction = '->' if Q_sm3s >= 0 else '<-'

        # Actual velocity at average pipe pressure
        n_from = node_map[pipe.from_node]
        n_to   = node_map[pipe.to_node]
        P_from = n_from.pressure
        P_to   = n_to.pressure
        P_avg  = (P_from + P_to) / 2.0

        # Create a local gas property at average pressure for velocity calc
        Q_act  = sm3s_to_m3s(Q_abs, P_avg, gas.T, gas.Z)
        v      = Q_act / pipe.area if pipe.area > 0 else 0.0

        # Reynolds number
        Re = compute_reynolds(Q_sm3s, pipe, gas)

        # Friction factor
        f  = swamee_jain(Re, pipe.roughness, pipe.diameter)

        # Pressure drop [Pa and mbar]
        dP_Pa   = P_from - P_to              # Pa  (positive = from→to)
        dP_mbar = dP_Pa / 100.0             # mbar

        # Pressure gradient [mbar/m]
        grad_mbar_m = abs(dP_mbar) / pipe.length if pipe.length > 0 else 0.0

        # Erosional velocity at average pressure
        from gas_properties import compute_gas_properties
        gas_avg = compute_gas_properties(P_avg, gas.T, gas.SG)
        v_erosional = erosional_velocity(P_avg, gas_avg)

        # Loading ratio = v / v_erosional × 100  [%]
        loading_pct = (v / v_erosional * 100.0) if v_erosional > 0 else 0.0

        # Status flags
        flags = []
        if v > v_erosional:
            flags.append('[!] EROSIONAL')
        if grad_mbar_m > MAX_GRAD_MBAR_M:
            flags.append('[!] HIGH GRAD')
        if loading_pct > CRIT_LOADING_PCT:
            flags.append('[!] CRITICAL')
        elif loading_pct > WARN_LOADING_PCT:
            flags.append('[!] LOADED')
        if Q_sm3s < 0:
            flags.append('[REV] REVERSED')
        status = ' | '.join(flags) if flags else '[OK]'

        rows.append({
            'Pipe_ID':      pipe.id,
            'From':         pipe.from_node,
            'To':           pipe.to_node,
            'Q_sm3s':       round(Q_sm3s, 8),
            'Q_sm3h':       round(Q_sm3h, 3),
            'Direction':    direction,
            'v_ms':         round(v, 4),
            'v_erosional':  round(v_erosional, 3),
            'Re':           int(Re),
            'f':            round(f, 6),
            'dP_Pa':        round(dP_Pa, 2),
            'dP_mbar':      round(dP_mbar, 3),
            'Grad_mbar_m':  round(grad_mbar_m, 5),
            'Loading_pct':  round(loading_pct, 2),
            'Status':       status,
        })

    return pd.DataFrame(rows)


def compute_node_results(nodes: List[Node],
                         result) -> pd.DataFrame:
    """
    Compute per-node results including pressure margin.

    Parameters
    ----------
    nodes  : list of Node  (with solved .pressure)
    result : SolverResult

    Returns
    -------
    pd.DataFrame  one row per node
    """
    rows = []
    for i, n in enumerate(nodes):
        P_kpag    = (n.pressure - P_STD) / 1000.0   # kPa(g)
        min_kpag  = (n.min_pressure - P_STD) / 1000.0
        margin    = P_kpag - min_kpag

        if n.is_source():
            status = 'SOURCE (fixed P)'
        elif margin < 0:
            status = f'[FAIL] BELOW MIN by {abs(margin):.1f} kPa'
        elif margin < 10:
            status = f'[!] LOW MARGIN ({margin:.1f} kPa)'
        else:
            status = f'[OK] ({margin:.1f} kPa margin)'

        rows.append({
            'Node_ID':   n.id,
            'Label':     n.label,
            'Type':      n.node_type,
            'P_kPag':    round(P_kpag, 2),
            'MinP_kPag': round(min_kpag, 2),
            'Margin_kPa': round(margin, 2),
            'Status':    status,
        })

    return pd.DataFrame(rows)


def mass_balance(nodes: List[Node],
                 pipes: List[Pipe]) -> Dict:
    """
    Compute network-level mass balance.

    Returns
    -------
    dict with keys:
        total_supply_sm3h  : float
        total_demand_sm3h  : float
        imbalance_sm3h     : float
        balanced           : bool   (imbalance < 0.1 Sm³/h)
    """
    node_map = {n.id: n for n in nodes}

    # Supply = net outflow from source nodes = A[src,:] @ Q (signed)
    # Negative supply means that source is being backfed by the network.
    import numpy as np
    from network import build_incidence_matrix
    A       = build_incidence_matrix(nodes, pipes)
    src_idx = [i for i, n in enumerate(nodes) if n.is_source()]
    Q_arr   = np.array([p.flow for p in pipes])
    supply  = float(np.sum(A[src_idx, :] @ Q_arr)) * 3600
    demand  = sum(n.demand * 3600 for n in nodes if n.is_offtake())
    imbal   = supply - demand

    return {
        'total_supply_sm3h': round(supply, 3),
        'total_demand_sm3h': round(demand, 3),
        'imbalance_sm3h':    round(imbal, 4),
        'balanced':          abs(imbal) < 0.1,
    }


def print_summary(nodes:   List[Node],
                  pipes:   List[Pipe],
                  result,
                  gas:     GasProperties) -> None:
    """
    Print a formatted engineering summary to stdout.
    """
    pipe_df = compute_pipe_results(pipes, nodes, result, gas)
    node_df = compute_node_results(nodes, result)
    mb      = mass_balance(nodes, pipes)

    print("\n" + "="*72)
    print("  SOLVER RESULT SUMMARY")
    print("="*72)
    print(f"  Status     : {'[CONVERGED]' if result.converged else '[NOT CONVERGED]'}")
    print(f"  Iterations : {result.iterations}")
    print(f"  max|F|     : {result.final_residual:.3e} Sm³/s")

    print("\n  -- NODE PRESSURES ----------------------------------------------")
    print(f"  {'Node':<8} {'Label':<16} {'Type':<10} {'P [kPa(g)]':>12} "
          f"{'Min P':>8} {'Margin':>8}  Status")
    print(f"  {'-'*72}")
    for _, row in node_df.iterrows():
        print(f"  {row['Node_ID']:<8} {row['Label']:<16} {row['Type']:<10} "
              f"{row['P_kPag']:>12.1f} {row['MinP_kPag']:>8.1f} "
              f"{row['Margin_kPa']:>8.1f}  {row['Status']}")

    print("\n  -- PIPE FLOWS --------------------------------------------------")
    print(f"  {'Pipe':<8} {'Route':<20} {'Q [Sm³/h]':>12} {'v [m/s]':>9} "
          f"{'Re':>10} {'f':>8} {'Grad[mb/m]':>11}  Status")
    print(f"  {'-'*72}")
    for _, row in pipe_df.iterrows():
        route = f"{row['From']}->{row['To']}"
        print(f"  {row['Pipe_ID']:<8} {route:<20} {row['Q_sm3h']:>12.2f} "
              f"{row['v_ms']:>9.3f} {row['Re']:>10,} {row['f']:>8.5f} "
              f"{row['Grad_mbar_m']:>11.4f}  {row['Status']}")

    print("\n  -- MASS BALANCE ------------------------------------------------")
    print(f"  Total supply : {mb['total_supply_sm3h']:>10.3f} Sm³/h")
    print(f"  Total demand : {mb['total_demand_sm3h']:>10.3f} Sm³/h")
    print(f"  Imbalance    : {mb['imbalance_sm3h']:>10.4f} Sm³/h  "
          f"({'[OK]' if mb['balanced'] else '[CHECK]'})")
    print("="*72 + "\n")


def generate_pressure_map(nodes: List[Node],
                          pipes: List[Pipe],
                          output_path: str = "network_pressure_map.html") -> None:
    """
    Generate an interactive Plotly pressure map (HTML file).

    Requires plotly to be installed: pip install plotly

    Parameters
    ----------
    nodes       : list of Node  (with GIS coordinates and solved pressures)
    pipes       : list of Pipe
    output_path : str  path to write the HTML file
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        return

    node_map = {n.id: n for n in nodes}

    # Node pressures in kPa(g) for colour scale
    P_kpag   = [(n.pressure - P_STD) / 1000.0 for n in nodes]
    P_min    = min(P_kpag)
    P_max    = max(P_kpag)

    # Pipe traces
    pipe_traces = []
    for pipe in pipes:
        nf = node_map[pipe.from_node]
        nt = node_map[pipe.to_node]
        pipe_traces.append(go.Scatter(
            x=[nf.x_coord, nt.x_coord, None],
            y=[nf.y_coord, nt.y_coord, None],
            mode='lines',
            line=dict(color='steelblue', width=3),
            hoverinfo='skip',
            showlegend=False,
        ))

    # Node trace
    color_map = {'source': 'gold', 'junction': 'cornflowerblue', 'offtake': 'mediumseagreen'}
    sym_map   = {'source': 'diamond', 'junction': 'circle', 'offtake': 'square'}

    for ntype in ['source', 'junction', 'offtake']:
        subset = [n for n in nodes if n.node_type == ntype]
        if not subset:
            continue
        pipe_traces.append(go.Scatter(
            x=[n.x_coord for n in subset],
            y=[n.y_coord for n in subset],
            mode='markers+text',
            marker=dict(
                symbol=sym_map[ntype],
                size=16,
                color=[(n.pressure - P_STD)/1000.0 for n in subset],
                colorscale='RdYlGn',
                cmin=P_min, cmax=P_max,
                showscale=(ntype == 'offtake'),
                colorbar=dict(title='Pressure<br>[kPa(g)]') if ntype == 'offtake' else None,
                line=dict(color=color_map[ntype], width=2),
            ),
            text=[f"{n.id}<br>{n.label}" for n in subset],
            textposition='top center',
            name=ntype.capitalize(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "P = %{marker.color:.1f} kPa(g)<br>"
                "Type: " + ntype + "<extra></extra>"
            ),
        ))

    fig = go.Figure(data=pipe_traces)
    fig.update_layout(
        title="PE80 Network — Nodal Pressure Map",
        xaxis_title="Easting [m]",
        yaxis_title="Northing [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=700,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )
    fig.write_html(output_path)
    print(f"Pressure map saved to: {output_path}")


def generate_flow_map(nodes:   List[Node],
                      pipes:   List[Pipe],
                      result,
                      gas:     'GasProperties',
                      output_path: str = "network_flow_map.html") -> None:
    """
    Generate an interactive Plotly flow map (HTML).

    Pipes are drawn with width proportional to |Q|, coloured by status
    (normal / reversed / high-gradient / erosional), with directional
    arrowheads at the pipe midpoint.  Nodes are coloured by pressure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        return

    node_map  = {n.id: n for n in nodes}
    pipe_df   = compute_pipe_results(pipes, nodes, result, gas)
    pipe_info = {row['Pipe_ID']: row for _, row in pipe_df.iterrows()}

    flows_abs = [abs(pipe_info[p.id]['Q_sm3h']) for p in pipes]
    max_flow  = max(flows_abs) if flows_abs else 1.0

    P_kpag = [(n.pressure - P_STD) / 1000.0 for n in nodes]
    P_min  = min(P_kpag)
    P_max  = max(P_kpag)

    STATUS_COLOR = {
        '[OK]':      'steelblue',
        'REVERSED':  'darkorange',
        'HIGH GRAD': 'crimson',
        'EROSIONAL': 'darkred',
        'CRITICAL':  'darkred',
        'LOADED':    'goldenrod',
    }

    def _pipe_color(status: str) -> str:
        for key, col in STATUS_COLOR.items():
            if key in status:
                return col
        return 'steelblue'

    traces = []

    # ── Invisible midpoint scatter for hover ─────────────────────────────
    mid_x, mid_y, hover_text = [], [], []
    for pipe in pipes:
        nf  = node_map[pipe.from_node]
        nt  = node_map[pipe.to_node]
        inf = pipe_info[pipe.id]
        mid_x.append((nf.x_coord + nt.x_coord) / 2)
        mid_y.append((nf.y_coord + nt.y_coord) / 2)
        hover_text.append(
            f"<b>{pipe.id}</b>  {nf.id} {inf['Direction']} {nt.id}<br>"
            f"Q  = {inf['Q_sm3h']:+.1f} Sm³/h<br>"
            f"v  = {inf['v_ms']:.3f} m/s  (v_ero = {inf['v_erosional']:.2f})<br>"
            f"Re = {inf['Re']:,}<br>"
            f"f  = {inf['f']:.5f}<br>"
            f"ΔP = {inf['dP_mbar']:.2f} mbar<br>"
            f"∇P = {inf['Grad_mbar_m']:.4f} mbar/m<br>"
            f"Load = {inf['Loading_pct']:.1f}%<br>"
            f"<b>{inf['Status']}</b>"
        )

    traces.append(go.Scatter(
        x=mid_x, y=mid_y,
        mode='markers',
        marker=dict(size=14, color='rgba(0,0,0,0)'),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False,
    ))

    # ── Pipe lines + direction arrows ─────────────────────────────────────
    annotations = []
    for pipe in pipes:
        nf  = node_map[pipe.from_node]
        nt  = node_map[pipe.to_node]
        inf = pipe_info[pipe.id]
        Q   = inf['Q_sm3h']
        col = _pipe_color(inf['Status'])
        lw  = max(2.0, min(12.0, 2.0 + 10.0 * abs(Q) / max_flow))

        traces.append(go.Scatter(
            x=[nf.x_coord, nt.x_coord, None],
            y=[nf.y_coord, nt.y_coord, None],
            mode='lines',
            line=dict(color=col, width=lw),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Arrow placed at 40% → 62% of segment (in actual flow direction)
        dx = nt.x_coord - nf.x_coord
        dy = nt.y_coord - nf.y_coord
        if Q >= 0:
            ax, ay = nf.x_coord + 0.38*dx, nf.y_coord + 0.38*dy
            ex, ey = nf.x_coord + 0.62*dx, nf.y_coord + 0.62*dy
        else:
            ax, ay = nf.x_coord + 0.62*dx, nf.y_coord + 0.62*dy
            ex, ey = nf.x_coord + 0.38*dx, nf.y_coord + 0.38*dy

        annotations.append(dict(
            x=ex, y=ey, ax=ax, ay=ay,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1.5, arrowwidth=max(1.5, lw * 0.5),
            arrowcolor=col,
            showarrow=True, text='',
        ))

    # ── Pipe ID labels at midpoint ────────────────────────────────────────
    traces.append(go.Scatter(
        x=mid_x, y=mid_y,
        mode='text',
        text=[pipe_info[p.id]['Pipe_ID'] for p in pipes],
        textfont=dict(size=9, color='dimgray'),
        hoverinfo='skip',
        showlegend=False,
    ))

    # ── Node markers coloured by pressure ────────────────────────────────
    color_map = {'source': 'gold', 'junction': 'cornflowerblue', 'offtake': 'mediumseagreen'}
    sym_map   = {'source': 'diamond', 'junction': 'circle', 'offtake': 'square'}

    for ntype in ['source', 'junction', 'offtake']:
        subset = [n for n in nodes if n.node_type == ntype]
        if not subset:
            continue
        P_vals = [(n.pressure - P_STD) / 1000.0 for n in subset]
        demand_vals = [n.demand * 3600 for n in subset]
        traces.append(go.Scatter(
            x=[n.x_coord for n in subset],
            y=[n.y_coord for n in subset],
            mode='markers+text',
            marker=dict(
                symbol=sym_map[ntype],
                size=18,
                color=P_vals,
                colorscale='RdYlGn',
                cmin=P_min, cmax=P_max,
                showscale=(ntype == 'offtake'),
                colorbar=dict(title='Pressure<br>[kPa(g)]', x=1.02) if ntype == 'offtake' else None,
                line=dict(color=color_map[ntype], width=2),
            ),
            text=[f"{n.id}<br>{n.label}" for n in subset],
            textposition='top center',
            name=ntype.capitalize(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "P = %{marker.color:.1f} kPa(g)<br>"
                f"Demand = " + "%{customdata:.0f} Sm³/h<br>" +
                "Type: " + ntype + "<extra></extra>"
            ),
            customdata=demand_vals,
        ))

    # ── Legend items for pipe status colours ─────────────────────────────
    legend_items = [
        ('Normal flow',      'steelblue'),
        ('Reversed flow',    'darkorange'),
        ('High gradient',    'crimson'),
        ('Erosional/Crit.',  'darkred'),
    ]
    for label, col in legend_items:
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=col, width=4),
            name=label,
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        annotations=annotations,
        title="PE80 Network — Flow Map  (width ∝ |Q|, arrows show direction)",
        xaxis_title="Easting [m]",
        yaxis_title="Northing [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=750,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1),
    )
    fig.write_html(output_path)
    print(f"Flow map saved to: {output_path}")
