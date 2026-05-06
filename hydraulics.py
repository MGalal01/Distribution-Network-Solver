"""
hydraulics.py — Pipe conductance, flow equation, friction factor (Swamee-Jain).

Flow equation (P² quadratic form — mandatory above 75 mbar gauge):
    Q_j [Sm³/s] = K_j × sign(Φ_from - Φ_to) × sqrt(|Φ_from - Φ_to|)
    where Φ_i = P_i²  (node potential [Pa²])

Pipe conductance (derived from Darcy-Weisbach + ideal-gas continuity):
    K_j = A_j × T_std/P_std × sqrt(D_j × R / (Z × T × f_j × L_j × MW))

This form is P_avg-independent — K is a function of pipe geometry and
gas state only, not of the local average pressure.

All units SI (Pa, m, Sm³/s, kg/m³, Pa·s).
"""

import math
import numpy as np
from typing import List

from network        import Node, Pipe
from gas_properties import GasProperties, P_STD, T_STD, R_UNIVERSAL

EPS_PHI = 1e4    # Pa²  — Jacobian regularisation (≈ 0.05 Pa diff at 4 bar)
EPS_RE  = 1.0    # minimum Reynolds number
EPS_Q   = 1e-12  # Sm³/s — near-zero flow floor


def swamee_jain(Re: float, roughness: float, diameter: float) -> float:
    """
    Swamee-Jain (1976) explicit friction factor.

    Laminar:    f = 64/Re          (Re < 2300)
    Turbulent:  f = 0.25/[log10(ε/3.7D + 5.74/Re^0.9)]²
    Transition: linear interpolation 2300–4000
    """
    Re = max(Re, EPS_RE)
    if Re < 2300:
        return 64.0 / Re
    eps_D  = roughness / diameter
    f_turb = 0.25 / (math.log10(eps_D / 3.7 + 5.74 / Re**0.9))**2
    if Re >= 4000:
        return f_turb
    f_lam = 64.0 / 2300.0
    alpha  = (Re - 2300.0) / 1700.0
    return f_lam + alpha * (f_turb - f_lam)


def compute_reynolds(Q_sm3s: float, pipe: Pipe,
                     gas: GasProperties) -> float:
    """Reynolds number using actual velocity at flowing conditions."""
    Q_abs = abs(Q_sm3s)
    if Q_abs < EPS_Q:
        return EPS_RE
    from gas_properties import sm3s_to_m3s
    Q_act = sm3s_to_m3s(Q_abs, gas.P_abs, gas.T, gas.Z)
    v     = Q_act / pipe.area
    return max(gas.rho * v * pipe.diameter / gas.mu, EPS_RE)


def compute_pipe_conductance(pipe: Pipe, f: float,
                              gas: GasProperties) -> float:
    """
    Pipe conductance K [Sm³/(s·Pa)] for the quadratic P² flow equation.

    Derivation:
        From Darcy-Weisbach with Q_act = Q_std × (P_std/P_avg) × (T/T_std) × Z:
            ΔP = f L ρ (Q_std × factor)² / (2 D A²)
        With ΔP ≈ (P₁²-P₂²)/(2P_avg) and ρ = P_avg MW/(Z R T):
            P₁²-P₂² = f L Q_std² MW P_std² / (D A² Z R T × factor²)
        where factor = P_std/P_avg × T/T_std × Z cancels P_avg, giving:
            K = A T_std/P_std × sqrt(D R / (Z T f L MW))  ← P_avg independent
    """
    if f < 1e-9:
        f = 0.02
    MW = gas.MW   # kg/mol
    term = pipe.diameter * R_UNIVERSAL / (gas.Z * gas.T * f * pipe.length * MW)
    return pipe.area * (T_STD / P_STD) * math.sqrt(max(term, 0.0))


def compute_friction_factors(pipes: List[Pipe], flows: np.ndarray,
                              gas: GasProperties) -> np.ndarray:
    return np.array([swamee_jain(compute_reynolds(flows[j], p, gas),
                                 p.roughness, p.diameter)
                     for j, p in enumerate(pipes)])


def delta_phi(Phi_all: np.ndarray, A: np.ndarray) -> np.ndarray:
    """ΔΦ_j = Φ_from - Φ_to = (A^T @ Φ)_j  where Φ_i = P_i²"""
    return A.T @ Phi_all


def compute_flows_phi(dphi: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Q_j = sign(ΔΦ_j) × K_j × |ΔΦ_j|^0.5"""
    return np.sign(dphi) * K * np.sqrt(np.abs(dphi))


def compute_residual(Q: np.ndarray, A_u: np.ndarray,
                     D_u: np.ndarray) -> np.ndarray:
    """F_i = (A_u @ Q)_i - D_u_i  →  0 at convergence"""
    return A_u @ Q + D_u


def compute_jacobian_phi(A_u: np.ndarray, K: np.ndarray,
                          dphi: np.ndarray) -> np.ndarray:
    """
    Jacobian of F w.r.t. Φ_unknown.
    J = A_u @ diag(W) @ A_u^T   where W_jj = 0.5 K_j / (|ΔΦ_j|^0.5 + ε)
    Symmetric and positive definite for a connected network.
    """
    W_diag = 0.5 * K / (np.sqrt(np.abs(dphi)) + EPS_PHI)
    return A_u @ np.diag(W_diag) @ A_u.T


def erosional_velocity(P_abs: float, gas: GasProperties) -> float:
    """API RP 14E erosional velocity C/sqrt(ρ), C=122 SI."""
    return 122.0 / math.sqrt(max(gas.rho, 0.1))
