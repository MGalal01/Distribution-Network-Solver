"""
solver.py — Newton-Raphson solver in Φ = P² (node potential) space.

Working in Φ space avoids the chain-rule and gives a symmetric,
positive-definite Jacobian that is well-conditioned for connected networks.

Algorithm
---------
  1. Build incidence matrix A; partition into source / unknown nodes.
  2. Initialise Φ_unknown via BFS pressure propagation from sources
     (estimates ΔΦ per pipe from expected demand flow).
  3. Iterate Newton-Raphson in Φ space:
       a. Compute K (pipe conductances, updated friction factors)
       b. Compute ΔΦ, flows Q, continuity residual F
       c. Jacobian  J = A_u @ diag(W) @ A_u^T
       d. Solve J·δΦ = −F
       e. Backtracking line search
       f. Clamp Φ ≥ Φ_floor; restore fixed-source Φ values
  4. Post-convergence: recover P = sqrt(Φ); write back to nodes/pipes.
"""

import math
import numpy as np
import logging
from dataclasses import dataclass
from typing import List

from network    import Node, Pipe, build_incidence_matrix, partition_nodes
from hydraulics import (compute_pipe_conductance, delta_phi, compute_flows_phi,
                          compute_residual, compute_jacobian_phi,
                          compute_friction_factors, swamee_jain, EPS_PHI)
from gas_properties import GasProperties, P_STD

log = logging.getLogger(__name__)


@dataclass
class SolverResult:
    converged:      bool
    iterations:     int
    final_residual: float
    P_all:          np.ndarray   # Pa(a)
    Q:              np.ndarray   # Sm³/s
    f:              np.ndarray   # Darcy friction factors
    Re:             np.ndarray   # Reynolds numbers
    message:        str


# ─────────────────────────────────────────────────────────────────────────────
def _bfs_init_phi(nodes, pipes, src_idx, unk_idx, node_idx,
                  Phi_src, gas, p_floor):
    """
    Initialise Φ for unknown nodes via BFS from sources.

    For each pipe in BFS order, estimate ΔΦ from the cumulative downstream
    demand divided by an initial conductance (f = 0.02 conservative).
    Sets each unknown Φ = Φ_parent − ΔΦ_est, clamped to Φ_floor.
    """
    Phi_all = np.zeros(len(nodes))
    for i in src_idx:
        Phi_all[i] = nodes[i].pressure ** 2

    # Initial conductances with conservative friction
    f0 = np.array([swamee_jain(1e5, p.roughness, p.diameter) for p in pipes])
    K0 = np.array([compute_pipe_conductance(p, f0[j], gas)
                   for j, p in enumerate(pipes)])

    # Build adjacency with pipe index
    from collections import deque
    node_map  = {n.id: i for i, n in enumerate(nodes)}
    adj = {n.id: [] for n in nodes}   # node_id → [(neighbour_id, pipe_idx)]
    for j, p in enumerate(pipes):
        if p.status == 'open':
            adj[p.from_node].append((p.to_node, j,  1))   # +1: flow leaves from_node
            adj[p.to_node].append(  (p.from_node, j, -1))

    # Total demand downstream of each node (rough BFS-order estimate)
    total_demand = sum(n.demand for n in nodes if n.is_offtake())

    visited = set(n.id for n in nodes if n.is_source())
    queue   = deque(n.id for n in nodes if n.is_source())
    Phi_floor = p_floor ** 2

    while queue:
        cur_id = queue.popleft()
        cur_i  = node_map[cur_id]
        for (nb_id, j, sign) in adj[cur_id]:
            if nb_id in visited:
                continue
            nb_i = node_map[nb_id]
            if nodes[nb_i].is_source():
                Phi_all[nb_i] = nodes[nb_i].pressure ** 2
                visited.add(nb_id)
                queue.append(nb_id)
                continue

            # Estimate flow as fraction of total demand
            Q_est = max(nodes[nb_i].demand, total_demand * 0.05)
            K_j   = max(K0[j], 1e-15)
            dPhi  = (Q_est / K_j) ** 2
            Phi_all[nb_i] = max(Phi_all[cur_i] - dPhi, Phi_floor)
            visited.add(nb_id)
            queue.append(nb_id)

    # Any still-unvisited unknowns → set to min source Phi × 0.9
    min_src_phi = min(Phi_all[i] for i in src_idx)
    for i in unk_idx:
        if Phi_all[i] == 0:
            Phi_all[i] = max(min_src_phi * 0.9, Phi_floor)

    return Phi_all


# ─────────────────────────────────────────────────────────────────────────────
def solve_network(
    nodes:       List[Node],
    pipes:       List[Pipe],
    gas:         GasProperties,
    tol:         float = 1e-6,
    max_iter:    int   = 150,
    damping:     float = 1.0,
    min_damping: float = 0.001,
    p_floor:     float = P_STD,
    verbose:     bool  = True,
) -> SolverResult:
    """
    Newton-Raphson solver in Φ = P² space.

    Parameters
    ----------
    nodes       : list of Node  (sources carry fixed pressures)
    pipes       : list of Pipe  (open pipes only)
    gas         : GasProperties at representative network conditions
    tol         : convergence tolerance [Sm³/s]
    max_iter    : maximum iterations
    damping     : initial Newton step size
    min_damping : minimum after backtracking
    p_floor     : minimum allowable pressure [Pa(a)]
    verbose     : print convergence table
    """
    n_nodes = len(nodes)
    n_pipes = len(pipes)
    node_idx = {n.id: i for i, n in enumerate(nodes)}

    if n_nodes == 0 or n_pipes == 0:
        return SolverResult(False, 0, float('inf'),
                            np.zeros(0), np.zeros(0),
                            np.zeros(0), np.zeros(0), "Empty network")

    A = build_incidence_matrix(nodes, pipes)
    src_idx, unk_idx = partition_nodes(nodes)

    if not src_idx:
        return SolverResult(False, 0, float('inf'),
                            np.zeros(n_nodes), np.zeros(n_pipes),
                            np.zeros(n_pipes), np.zeros(n_pipes),
                            "No source nodes found")

    A_u   = A[unk_idx, :]            # (n_u × n_pipes)
    n_u   = len(unk_idx)
    D_u   = np.array([nodes[i].demand for i in unk_idx])
    Phi_floor = p_floor ** 2

    # ── Smart initial Φ via BFS ───────────────────────────────────────────────
    Phi_all = _bfs_init_phi(nodes, pipes, src_idx, unk_idx,
                             node_idx, None, gas, p_floor)

    # ── Initial friction factors (fully-turbulent estimate) ───────────────────
    f = np.array([swamee_jain(1e5, p.roughness, p.diameter) for p in pipes])

    if verbose:
        print(f"\n{'='*64}")
        print(f"  PE80 Network Solver  (N-R in Phi=P^2 space)")
        print(f"  Nodes: {n_nodes}  Pipes: {n_pipes}  "
              f"Unknowns: {n_u}  tol: {tol:.1e} Sm³/s")
        print(f"{'='*64}")
        print(f"  {'Iter':>4}  {'max|F| [Sm3/s]':>18}  {'Step a':>8}")
        print(f"  {'-'*36}")

    res_norm = float('inf')
    alpha    = damping

    for iteration in range(max_iter):

        # 1. Conductances from current friction factors
        K = np.array([compute_pipe_conductance(pipes[j], f[j], gas)
                      for j in range(n_pipes)])

        # 2. ΔΦ, flows, residual
        dphi = delta_phi(Phi_all, A)
        Q    = compute_flows_phi(dphi, K)
        F    = compute_residual(Q, A_u, D_u)
        res_norm = float(np.max(np.abs(F)))

        if verbose:
            print(f"  {iteration:>4}  {res_norm:>18.6e}  "
                  f"{'---':>8}" if iteration == 0
                  else f"  {iteration:>4}  {res_norm:>18.6e}  {alpha:>8.4f}")

        if res_norm < tol:
            if verbose:
                print(f"  {'-'*36}")
                print(f"  Converged in {iteration} iter. "
                      f"max|F| = {res_norm:.3e} Sm³/s\n{'='*64}\n")
            break

        # 3. Jacobian in Φ space (symmetric, positive-definite)
        J = compute_jacobian_phi(A_u, K, dphi)

        # Regularise if near-singular (isolated node guard)
        J += np.eye(n_u) * 1e-30

        # 4. Solve J·δΦ = −F
        try:
            delta_Phi_u = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return SolverResult(False, iteration, res_norm,
                                np.sqrt(np.maximum(Phi_all, 0)),
                                Q, f, np.zeros(n_pipes),
                                "Singular Jacobian -- network disconnected?")

        # 5. Backtracking line search
        alpha = damping
        for _ in range(16):
            Phi_trial = Phi_all.copy()
            for k, idx in enumerate(unk_idx):
                Phi_trial[idx] += alpha * delta_Phi_u[k]

            # Clamp: Φ ≥ floor; restore fixed sources
            Phi_trial = np.maximum(Phi_trial, Phi_floor)
            for i in src_idx:
                Phi_trial[i] = nodes[i].pressure ** 2

            dphi_t  = delta_phi(Phi_trial, A)
            Q_t     = compute_flows_phi(dphi_t, K)
            F_t     = compute_residual(Q_t, A_u, D_u)
            res_t   = float(np.max(np.abs(F_t)))

            if res_t < res_norm or alpha <= min_damping:
                break
            alpha *= 0.5

        Phi_all = Phi_trial

        # 6. Update friction factors
        f = compute_friction_factors(pipes, Q, gas)

    else:
        if verbose:
            print(f"  WARNING: max_iter={max_iter} reached. "
                  f"max|F|={res_norm:.3e} Sm³/s")

    # ── Final quantities ──────────────────────────────────────────────────────
    K    = np.array([compute_pipe_conductance(pipes[j], f[j], gas)
                     for j in range(n_pipes)])
    dphi = delta_phi(Phi_all, A)
    Q    = compute_flows_phi(dphi, K)

    from hydraulics import compute_reynolds
    Re   = np.array([compute_reynolds(Q[j], pipes[j], gas)
                     for j in range(n_pipes)])

    P_all = np.sqrt(np.maximum(Phi_all, 0.0))

    # Write back to objects
    for j, pipe in enumerate(pipes):
        pipe.flow = float(Q[j])
    for i, node in enumerate(nodes):
        node.pressure = float(P_all[i])

    converged = res_norm < tol
    msg = (f"Converged in {iteration+1} iter, max|F|={res_norm:.2e} Sm³/s"
           if converged else
           f"Not converged after {max_iter} iter, max|F|={res_norm:.2e} Sm³/s")

    return SolverResult(converged=converged, iterations=iteration+1,
                        final_residual=res_norm, P_all=P_all,
                        Q=Q, f=f, Re=Re, message=msg)
