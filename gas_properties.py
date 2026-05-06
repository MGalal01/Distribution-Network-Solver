"""
gas_properties.py  — Z-factor (Papay), density, viscosity (Lee-Kesler).
All SI units.  T in K internally; Lee-Kesler converted to Rankine internally.
"""
import math
from dataclasses import dataclass

P_STD       = 101_325.0   # Pa(a)
T_STD       = 288.75      # K  (15.6 °C)
R_UNIVERSAL = 8.314       # J/(mol·K)


@dataclass
class GasProperties:
    P_abs: float; T: float; Z: float
    rho:   float; mu: float; MW: float; SG: float


def z_papay(P_abs: float, T: float, SG: float = 0.62) -> float:
    """Papay (1968) Z-factor.  Ppc in psia → converted to Pa."""
    Tpc = 93.3 + 180.6 * SG
    Ppc = (677.0 + 15.0 * SG) * 6894.76   # psia → Pa  ← BUG FIX
    Tr  = T    / Tpc
    Pr  = P_abs / Ppc
    Z   = (1.0
           - (3.52 * Pr)    / (10 ** (0.9813 * Tr))
           + (0.274 * Pr**2) / (10 ** (0.8157 * Tr)))
    return max(Z, 0.5)


def gas_density(P_abs: float, T: float,
                SG: float = 0.62, Z: float = None) -> float:
    MW_air = 0.028964
    MW     = SG * MW_air
    if Z is None:
        Z = z_papay(P_abs, T, SG)
    return (P_abs * MW) / (Z * R_UNIVERSAL * T)


def gas_viscosity_lee_kesler(rho: float, T: float,
                              MW_g_mol: float = 17.5) -> float:
    """Lee & Kesler (1975).  T must be in Rankine (K × 1.8).  ← BUG FIX"""
    T_R     = T * 1.8                # K → °R
    rho_gcc = rho / 1000.0           # kg/m³ → g/cm³
    K  = (9.4 + 0.02*MW_g_mol) * T_R**1.5 / (209.0 + 19.0*MW_g_mol + T_R)
    X  = 3.5 + 986.0/T_R + 0.01*MW_g_mol
    Y  = 2.4 - 0.2*X
    mu_cp = 1e-4 * K * math.exp(X * rho_gcc**Y)
    return mu_cp * 1e-3   # Pa·s


def compute_gas_properties(P_abs: float, T: float,
                            SG: float = 0.62,
                            MW_g_mol: float = None) -> GasProperties:
    if MW_g_mol is None:
        MW_g_mol = SG * 28.964
    Z   = z_papay(P_abs, T, SG)
    rho = gas_density(P_abs, T, SG, Z)
    mu  = gas_viscosity_lee_kesler(rho, T, MW_g_mol)
    return GasProperties(P_abs=P_abs, T=T, Z=Z, rho=rho,
                         mu=mu, MW=MW_g_mol*1e-3, SG=SG)


def sm3s_to_m3s(Q, P_abs, T, Z, P_std=P_STD, T_std=T_STD):
    return Q * (P_std/P_abs) * (T/T_std) * Z

def m3s_to_sm3s(Q, P_abs, T, Z, P_std=P_STD, T_std=T_STD):
    return Q * (P_abs/P_std) * (T_std/T) / Z
