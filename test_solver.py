"""
tests/test_solver.py — Validation test suite for the PE80 hydraulic solver.
Run:  python -m pytest tests/ -v
"""
import math, sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gas_properties import (z_papay, gas_density, gas_viscosity_lee_kesler,
                                  compute_gas_properties, P_STD, T_STD)
from src.network   import (Node, Pipe, build_incidence_matrix,
                            check_connectivity, partition_nodes)
from src.hydraulics import (swamee_jain, compute_reynolds,
                             compute_pipe_conductance, delta_phi,
                             compute_flows_phi, compute_residual,
                             compute_jacobian_phi)
from src.solver    import solve_network
from src.validate  import validate_network


# ═══════════════════════════════════════════════════════════════
# GAS PROPERTIES
# ═══════════════════════════════════════════════════════════════

class TestGasProperties:

    def test_z_atm(self):
        """Z ≈ 1 at atmospheric (Papay gives ~0.997)."""
        Z = z_papay(P_STD, 293.15, SG=0.62)
        assert 0.97 < Z < 1.02, f"Z={Z:.4f}"

    def test_z_4bar(self):
        """Z at 4 bar(g): Papay with correct Ppc gives ~0.985."""
        Z = z_papay(4e5 + P_STD, 293.15, SG=0.62)
        assert 0.97 < Z < 0.995, f"Z={Z:.4f}"

    def test_density_increases_with_pressure(self):
        rho1 = gas_density(P_STD,       293.15, 0.62)
        rho2 = gas_density(4e5 + P_STD, 293.15, 0.62)
        assert rho2 > rho1

    def test_density_at_std(self):
        """ρ at standard conditions ≈ 0.72–0.76 kg/m³ for SG=0.62."""
        rho = gas_density(P_STD, T_STD, 0.62)
        assert 0.70 < rho < 0.80, f"rho={rho:.4f}"

    def test_viscosity_magnitude(self):
        """μ ≈ 1.1e-5 Pa·s for nat gas at 20°C.  Lee-Kesler needs T in °R."""
        rho = gas_density(P_STD, 293.15, 0.62)
        mu  = gas_viscosity_lee_kesler(rho, 293.15)
        assert 8e-6 < mu < 2e-5, f"mu={mu:.2e}"

    def test_gas_props_dataclass(self):
        gas = compute_gas_properties(4e5 + P_STD, 293.15)
        assert gas.Z > 0 and gas.rho > 0 and gas.mu > 0


# ═══════════════════════════════════════════════════════════════
# FRICTION FACTOR
# ═══════════════════════════════════════════════════════════════

class TestFrictionFactor:

    def test_laminar(self):
        f = swamee_jain(1000, 7e-6, 0.1)
        assert abs(f - 64/1000) < 1e-6

    def test_turbulent_smooth(self):
        f = swamee_jain(1e6, 7e-6, 0.1)
        assert 0.010 < f < 0.016

    def test_f_decreases_with_Re(self):
        f1 = swamee_jain(1e4, 7e-6, 0.1)
        f2 = swamee_jain(1e5, 7e-6, 0.1)
        f3 = swamee_jain(1e6, 7e-6, 0.1)
        assert f1 > f2 > f3

    def test_rougher_pipe_higher_f(self):
        f_pe = swamee_jain(1e5, 7e-6,  0.1)
        f_st = swamee_jain(1e5, 46e-6, 0.1)
        assert f_st > f_pe


# ═══════════════════════════════════════════════════════════════
# TOPOLOGY
# ═══════════════════════════════════════════════════════════════

class TestTopology:

    def _net(self):
        nodes = [Node('N1','Src','source',pressure=5e5+P_STD),
                 Node('N2','Mid','junction'),
                 Node('N3','End','offtake',demand=0.10)]
        pipes = [Pipe('P1','N1','N2',500,0.1),
                 Pipe('P2','N2','N3',300,0.08)]
        return nodes, pipes

    def test_incidence_shape(self):
        n, p = self._net()
        assert build_incidence_matrix(n, p).shape == (3, 2)

    def test_incidence_values(self):
        n, p = self._net()
        A = build_incidence_matrix(n, p)
        assert A[0,0]==+1 and A[1,0]==-1
        assert A[1,1]==+1 and A[2,1]==-1

    def test_column_sums_zero(self):
        n, p = self._net()
        A = build_incidence_matrix(n, p)
        assert np.allclose(A.sum(axis=0), 0)

    def test_connected(self):
        n, p = self._net()
        assert check_connectivity(n, p)['connected']

    def test_disconnected(self):
        n, p = self._net()
        assert not check_connectivity(n, [])['connected']

    def test_partition(self):
        n, p = self._net()
        src, unk = partition_nodes(n)
        assert src == [0] and unk == [1, 2]


# ═══════════════════════════════════════════════════════════════
# SINGLE PIPE
# ═══════════════════════════════════════════════════════════════

class TestSinglePipe:
    """D=97mm, L=1000m, Q=0.10 Sm³/s (realistic for this size)."""
    P1  = 400e3 + P_STD
    D   = 0.09706; L = 1000.0; eps = 7e-6

    def _setup(self, demand=0.10):
        nodes = [Node('N1','PRS','source', pressure=self.P1),
                 Node('N2','End','offtake',demand=demand)]
        pipes = [Pipe('P1','N1','N2', self.L, self.D, self.eps)]
        gas   = compute_gas_properties(self.P1, 293.15, 0.62)
        return nodes, pipes, gas

    def test_conductance_positive(self):
        nodes, pipes, gas = self._setup()
        f = swamee_jain(5e4, self.eps, self.D)
        K = compute_pipe_conductance(pipes[0], f, gas)
        assert K > 0, f"K={K}"

    def test_converges(self):
        nodes, pipes, gas = self._setup(demand=0.10)
        r = solve_network(nodes, pipes, gas, tol=1e-8, verbose=False)
        assert r.converged, r.message

    def test_flow_direction(self):
        nodes, pipes, gas = self._setup(demand=0.10)
        r = solve_network(nodes, pipes, gas, tol=1e-8, verbose=False)
        assert r.Q[0] > 0

    def test_mass_balance(self):
        nodes, pipes, gas = self._setup(demand=0.10)
        r = solve_network(nodes, pipes, gas, tol=1e-8, verbose=False)
        assert abs(abs(r.Q[0]) - nodes[1].demand) < 1e-6


# ═══════════════════════════════════════════════════════════════
# SERIES PIPES
# ═══════════════════════════════════════════════════════════════

class TestSeriesPipes:
    """N1(src) → N2(junc) → N3(offtake), realistic 0.08 Sm³/s demand."""

    def _setup(self):
        P1 = 400e3 + P_STD
        nodes = [Node('N1','PRS','source',  pressure=P1),
                 Node('N2','Mid','junction'),
                 Node('N3','End','offtake', demand=0.08)]
        pipes = [Pipe('P1','N1','N2',600,0.12,7e-6),
                 Pipe('P2','N2','N3',400,0.09,7e-6)]
        gas = compute_gas_properties(P1, 293.15, 0.62)
        return nodes, pipes, gas

    def test_converges(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        assert r.converged, r.message

    def test_equal_flow_in_series(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        assert abs(abs(r.Q[0]) - abs(r.Q[1])) < 1e-7

    def test_pressure_decreases(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        assert r.P_all[1] < r.P_all[0]

    def test_mass_balance(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        supply = abs(r.Q[0]) * 3600
        demand = n[2].demand * 3600
        assert abs(supply - demand) < 0.001


# ═══════════════════════════════════════════════════════════════
# SIMPLE LOOP (4-node square)
# ═══════════════════════════════════════════════════════════════

class TestSimpleLoop:

    def _setup(self):
        P1 = 400e3 + P_STD
        nodes = [Node('N1','PRS', 'source',  pressure=P1),
                 Node('N2','TEE1','junction'),
                 Node('N3','TEE2','junction'),
                 Node('N4','End', 'offtake', demand=0.10)]
        pipes = [Pipe('P1','N1','N2',500,0.10,7e-6),
                 Pipe('P2','N2','N3',400,0.08,7e-6),
                 Pipe('P3','N3','N4',300,0.08,7e-6),
                 Pipe('P4','N1','N4',600,0.08,7e-6)]
        gas = compute_gas_properties(P1, 293.15, 0.62)
        return nodes, pipes, gas

    def test_converges(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        assert r.converged, r.message

    def test_node_continuity(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        A   = build_incidence_matrix(n, p)
        net = A @ r.Q
        assert abs(net[1]) < 1e-7, f"N2 imbalance: {net[1]:.2e}"
        assert abs(net[2]) < 1e-7, f"N3 imbalance: {net[2]:.2e}"

    def test_mass_balance(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        supply = (abs(r.Q[0]) + abs(r.Q[3])) * 3600
        demand = n[3].demand * 3600
        assert abs(supply - demand) < 0.01

    def test_pressure_loop_closure(self):
        n,p,g = self._setup()
        r = solve_network(n,p,g, tol=1e-8, verbose=False)
        P = r.P_all
        loop = (P[0]-P[1]) + (P[1]-P[2]) + (P[2]-P[3]) + (P[3]-P[0])
        assert abs(loop) < 1.0, f"Loop ΔP sum = {loop:.2f} Pa"


# ═══════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════

class TestValidation:

    def test_passes_valid(self):
        n = [Node('N1','PRS','source', pressure=4e5+P_STD),
             Node('N2','End','offtake',demand=0.10)]
        p = [Pipe('P1','N1','N2',500,0.1)]
        ok, _ = validate_network(n, p, verbose=False)
        assert ok

    def test_fails_no_source(self):
        n = [Node('N1','End','offtake',demand=0.10)]
        ok, errs = validate_network(n, [], verbose=False)
        assert not ok
        assert any('source' in e.lower() for e in errs)

    def test_fails_no_offtake(self):
        n = [Node('N1','PRS','source',pressure=4e5+P_STD)]
        ok, _ = validate_network(n, [], verbose=False)
        assert not ok

    def test_fails_disconnected(self):
        n = [Node('N1','PRS','source', pressure=4e5+P_STD),
             Node('N2','End','offtake',demand=0.10)]
        ok, errs = validate_network(n, [], verbose=False)
        assert not ok

    def test_fails_duplicate_pipe(self):
        n = [Node('N1','PRS','source', pressure=4e5+P_STD),
             Node('N2','End','offtake',demand=0.10)]
        p = [Pipe('P1','N1','N2',500,0.1), Pipe('P2','N1','N2',500,0.1)]
        ok, errs = validate_network(n, p, verbose=False)
        assert not ok
        assert any('duplicate' in e.lower() for e in errs)

    def test_fails_missing_node(self):
        """Missing node must fail without crashing (KeyError fixed)."""
        n = [Node('N1','PRS','source',pressure=4e5+P_STD)]
        p = [Pipe('P1','N1','N99',500,0.1)]
        ok, errs = validate_network(n, p, verbose=False)
        assert not ok


# ═══════════════════════════════════════════════════════════════
# TWO-SOURCE NETWORK (matches Excel template)
# ═══════════════════════════════════════════════════════════════

class TestTwoSources:

    def _setup(self):
        nodes = [
            Node('N001','PRS1',   'source',  pressure=400e3+P_STD),
            Node('N002','TEE1',   'junction'),
            Node('N003','TEE2',   'junction'),
            Node('N004','TEE3',   'junction'),
            Node('N005','Client1','offtake', demand=0.010),
            Node('N006','Client2','offtake', demand=0.015),
            Node('N007','Client3','offtake', demand=0.050),
            Node('N008','Client4','offtake', demand=0.008),
            Node('N009','PRS2',   'source',  pressure=375e3+P_STD),
            Node('N010','Client5','offtake', demand=0.025),
        ]
        pipes = [
            Pipe('P001','N001','N002',850, 0.1474,7e-6),
            Pipe('P002','N002','N003',420, 0.1153,7e-6),
            Pipe('P003','N002','N004',380, 0.1153,7e-6),
            Pipe('P004','N003','N005',310, 0.0829,7e-6),
            Pipe('P005','N003','N006',290, 0.1014,7e-6),
            Pipe('P006','N004','N007',260, 0.1014,7e-6),
            Pipe('P007','N004','N008',440, 0.0580,7e-6),
            Pipe('P008','N009','N004',620, 0.1153,7e-6),
            Pipe('P009','N004','N010',415, 0.0829,7e-6),
        ]
        gas = compute_gas_properties(4e5+P_STD, 293.15, 0.62)
        return nodes, pipes, gas

    def test_converges(self):
        n, p, g = self._setup()
        r = solve_network(n, p, g, tol=1e-7, verbose=False)
        assert r.converged, r.message

    def test_pressures_above_atmospheric(self):
        n, p, g = self._setup()
        r = solve_network(n, p, g, tol=1e-7, verbose=False)
        assert all(P > P_STD for P in r.P_all)

    def test_mass_balance(self):
        n, p, g = self._setup()
        r = solve_network(n, p, g, tol=1e-7, verbose=False)
        total_demand = sum(nd.demand for nd in n if nd.is_offtake()) * 3600
        # Supply = net outflow from all source nodes = -(A[src,:] @ Q)
        A = build_incidence_matrix(n, p)
        src_rows = [i for i, nd in enumerate(n) if nd.is_source()]
        total_supply = float(np.sum(A[src_rows, :] @ r.Q)) * 3600
        assert abs(total_supply - total_demand) < 0.5, \
            f"Supply {total_supply:.3f} ≠ demand {total_demand:.3f} Sm³/h"

    def test_source_pressures_fixed(self):
        n, p, g = self._setup()
        r = solve_network(n, p, g, tol=1e-7, verbose=False)
        assert abs(r.P_all[0] - (400e3+P_STD)) < 1.0
        assert abs(r.P_all[8] - (375e3+P_STD)) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
