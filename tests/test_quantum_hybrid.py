#!/usr/bin/env python3
"""
Tests for quantum-classical hybrid optimizer — data classes, QUBO formulation,
classical optimization, QAOA/VQE simulation, and hybrid optimization.
"""
import sys
import os
import math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from src.optimization.quantum_hybrid import (
    QuantumAlgorithm, AssetUniverse, QUBOFormulation,
    ClassicalOptimizer, QAOASimulator, VQESimulator,
    QuantumHybridOptimizer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_universe(n=5):
    """Create a small asset universe for testing."""
    return AssetUniverse.from_sample_data(n_assets=n)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestQuantumAlgorithm:
    """Test QuantumAlgorithm enum."""

    def test_values(self):
        assert QuantumAlgorithm.QAOA.value == 'qaoa'
        assert QuantumAlgorithm.VQE.value == 'vqe'
        assert QuantumAlgorithm.CLASSICAL.value == 'classical'


# ---------------------------------------------------------------------------
# AssetUniverse tests
# ---------------------------------------------------------------------------

class TestAssetUniverse:
    """Test AssetUniverse dataclass."""

    def test_from_sample_data(self):
        universe = AssetUniverse.from_sample_data(n_assets=5)
        assert len(universe.symbols) == 5
        assert len(universe.expected_returns) == 5
        assert universe.cov_matrix.shape == (5, 5)

    def test_symbols_default(self):
        universe = AssetUniverse.from_sample_data()
        assert universe.symbols[0] == 'SPY'
        assert 'GLD' in universe.symbols

    def test_cov_matrix_symmetric(self):
        universe = _make_universe(5)
        np.testing.assert_array_equal(universe.cov_matrix, universe.cov_matrix.T)

    def test_cov_matrix_positive_diagonal(self):
        universe = _make_universe(5)
        for i in range(5):
            assert universe.cov_matrix[i, i] > 0


# ---------------------------------------------------------------------------
# QUBOFormulation tests
# ---------------------------------------------------------------------------

class TestQUBOFormulation:
    """Test QUBOFormulation."""

    def test_decode_solution(self):
        Q = np.zeros((6, 6))
        qubo = QUBOFormulation(n_assets=2, n_bits_per_asset=3, Q=Q)
        # All bits set → equal weights
        binary = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        weights = qubo.decode_solution(binary)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01

    def test_decode_solution_single_asset(self):
        Q = np.zeros((6, 6))
        qubo = QUBOFormulation(n_assets=2, n_bits_per_asset=3, Q=Q)
        # Only first asset bits set
        binary = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        weights = qubo.decode_solution(binary)
        assert weights[0] > weights[1]

    def test_decode_solution_all_zeros(self):
        Q = np.zeros((6, 6))
        qubo = QUBOFormulation(n_assets=2, n_bits_per_asset=3, Q=Q)
        binary = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        weights = qubo.decode_solution(binary)
        assert all(w == 0 for w in weights)


# ---------------------------------------------------------------------------
# ClassicalOptimizer tests
# ---------------------------------------------------------------------------

class TestClassicalOptimizer:
    """Test ClassicalOptimizer."""

    def test_markowitz_returns_dict(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.markowitz_mean_variance()
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result

    def test_markowitz_weights_sum_to_one(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.markowitz_mean_variance()
        if 'error' not in result:
            total = sum(result['weights'])
            assert abs(total - 1.0) < 0.01

    def test_markowitz_sharpe_positive(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.markowitz_mean_variance()
        if 'error' not in result:
            assert result['sharpe_ratio'] > 0

    def test_risk_parity_returns_dict(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.risk_parity()
        assert 'weights' in result
        assert result['method'] == 'classical_risk_parity'

    def test_risk_parity_weights_sum_to_one(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.risk_parity()
        total = sum(result['weights'])
        assert abs(total - 1.0) < 0.01

    def test_risk_parity_all_positive(self):
        opt = ClassicalOptimizer(_make_universe(5))
        result = opt.risk_parity()
        for w in result['weights']:
            assert w >= 0

    def test_project_onto_simplex(self):
        opt = ClassicalOptimizer(_make_universe(3))
        v = np.array([0.5, 0.3, 0.2])
        projected = opt._project_onto_simplex(v)
        assert abs(sum(projected) - 1.0) < 0.01
        assert all(w >= 0 for w in projected)


# ---------------------------------------------------------------------------
# QAOASimulator tests
# ---------------------------------------------------------------------------

class TestQAOASimulator:
    """Test QAOASimulator."""

    def test_init(self):
        sim = QAOASimulator(n_qubits=6, p_layers=2)
        assert sim.n_qubits == 6
        assert sim.p == 2

    def test_optimize_returns_dict(self):
        sim = QAOASimulator(n_qubits=6, p_layers=1)
        Q = np.random.randn(6, 6) * 0.01
        Q = (Q + Q.T) / 2  # Symmetric
        result = sim.optimize(Q, max_iterations=10)
        assert 'solution' in result
        assert 'objective_value' in result
        assert 'parameters' in result

    def test_solution_binary(self):
        sim = QAOASimulator(n_qubits=6, p_layers=1)
        Q = np.eye(6) * 0.1
        result = sim.optimize(Q, max_iterations=10)
        for bit in result['solution']:
            assert bit in [0, 1]

    def test_layers_stored(self):
        sim = QAOASimulator(n_qubits=6, p_layers=2)
        Q = np.eye(6) * 0.1
        result = sim.optimize(Q, max_iterations=5)
        assert result['layers'] == 2


# ---------------------------------------------------------------------------
# VQESimulator tests
# ---------------------------------------------------------------------------

class TestVQESimulator:
    """Test VQESimulator."""

    def test_init(self):
        sim = VQESimulator(n_qubits=6, ansatz_depth=3)
        assert sim.n_qubits == 6
        assert sim.depth == 3

    def test_optimize_returns_dict(self):
        sim = VQESimulator(n_qubits=6, ansatz_depth=2)
        H = np.eye(6) * 0.1
        result = sim.optimize(H, max_iterations=10)
        assert 'solution' in result
        assert 'ground_state_energy' in result
        assert 'ansatz_depth' in result

    def test_solution_binary(self):
        sim = VQESimulator(n_qubits=6, ansatz_depth=2)
        H = np.eye(6) * 0.1
        result = sim.optimize(H, max_iterations=10)
        for bit in result['solution']:
            assert bit in [0, 1]

    def test_ansatz_returns_distribution(self):
        sim = VQESimulator(n_qubits=4, ansatz_depth=2)
        params = np.random.random(8) * 2 * np.pi
        probs = sim._hardware_efficient_ansatz(params)
        assert len(probs) == 2 ** 4  # 16 states
        assert abs(sum(probs) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# QuantumHybridOptimizer tests
# ---------------------------------------------------------------------------

class TestQuantumHybridOptimizer:
    """Test QuantumHybridOptimizer."""

    def test_init(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA)
        assert opt.universe == universe
        assert opt.algorithm == QuantumAlgorithm.QAOA

    def test_build_qubo(self):
        universe = _make_universe(3)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA, n_bits_per_asset=2)
        qubo = opt._build_qubo()
        assert isinstance(qubo, QUBOFormulation)
        assert qubo.n_assets == 3
        assert qubo.n_bits_per_asset == 2

    def test_classical_fallback(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.CLASSICAL)
        result = opt.optimize(use_quantum=False)
        assert result['success'] is True
        assert 'classical' in result['method']

    def test_quantum_too_many_assets_fallback(self):
        universe = _make_universe(10)  # Max sample size
        # Force >20 check by mocking symbols
        universe.symbols = ['A'] * 21
        universe.expected_returns = np.ones(21) * 0.10
        universe.cov_matrix = np.eye(21) * 0.0256
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA)
        result = opt.optimize(use_quantum=True)
        assert result['success'] is True
        assert 'classical' in result['method']

    def test_qaoa_optimization(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA, n_bits_per_asset=2)
        result = opt.optimize(use_quantum=True, quantum_p_layers=1)
        assert result['success'] is True
        assert 'hybrid' in result['method']

    def test_vqe_optimization(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.VQE, n_bits_per_asset=2)
        result = opt.optimize(use_quantum=True)
        assert result['success'] is True
        assert 'hybrid' in result['method']

    def test_result_has_metrics(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA, n_bits_per_asset=2)
        result = opt.optimize(use_quantum=True, quantum_p_layers=1)
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        assert 'weights' in result

    def test_weights_sum_to_one(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA, n_bits_per_asset=2)
        result = opt.optimize(use_quantum=True, quantum_p_layers=1)
        if 'weights' in result and sum(result['weights']) > 0:
            total = sum(result['weights'])
            assert abs(total - 1.0) < 0.05

    def test_cardinality_constraint(self):
        universe = _make_universe(5)
        opt = QuantumHybridOptimizer(
            universe, QuantumAlgorithm.QAOA,
            n_bits_per_asset=2, cardinality=2,
        )
        qubo = opt._build_qubo()
        assert qubo.n_assets == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
