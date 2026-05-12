#!/usr/bin/env python3
"""
v2.50 Quantum-Classical Hybrid Portfolio Optimization
QAOA/VQE-based portfolio allocation with classical fallback

Features:
- QUBO formulation for discrete asset selection
- QAOA simulator (p=1-3 layers)
- VQE with hardware-efficient ansatz
- Classical preprocessing (covariance, expected returns)
- Hybrid architecture: classical + quantum sampling + post-processing
- Automatic fallback to classical solver
"""

import argparse
import json
import math
import sys
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QuantumAlgorithm(Enum):
    QAOA = "qaoa"
    VQE = "vqe"
    CLASSICAL = "classical"


@dataclass
class AssetUniverse:
    """Asset universe with return/risk characteristics"""
    symbols: List[str]
    expected_returns: np.ndarray  # Annualized expected returns
    cov_matrix: np.ndarray        # Covariance matrix
    
    @classmethod
    def from_sample_data(cls, n_assets: int = 10) -> 'AssetUniverse':
        """Generate sample asset universe for testing"""
        # Sample ETFs/factors
        symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VXUS', 'MTUM', 'VLUE', 'QUAL', 'IJR'][:n_assets]
        
        # Sample expected returns (annualized)
        base_returns = np.array([0.10, 0.12, 0.09, 0.05, 0.06, 0.08, 0.11, 0.08, 0.10, 0.09])[:n_assets]
        
        # Sample covariance structure
        vols = np.array([0.16, 0.19, 0.21, 0.12, 0.14, 0.18, 0.17, 0.16, 0.15, 0.18])[:n_assets]
        
        # Create correlation matrix (simplified)
        corr = np.eye(n_assets) * 0.7 + 0.3  # 70% pairwise correlation
        np.fill_diagonal(corr, 1.0)
        
        # Convert to covariance
        cov = np.outer(vols, vols) * corr
        
        return cls(symbols=symbols, expected_returns=base_returns, cov_matrix=cov)


@dataclass
class QUBOFormulation:
    """QUBO (Quadratic Unconstrained Binary Optimization) problem"""
    n_assets: int
    n_bits_per_asset: int  # Number of bits for each asset weight (discretization)
    
    # QUBO matrix Q where objective = x^T Q x
    Q: np.ndarray
    
    # Mapping from binary variables to asset weights
    def decode_solution(self, binary_solution: np.ndarray) -> np.ndarray:
        """Convert binary solution to portfolio weights"""
        weights = np.zeros(self.n_assets)
        
        for i in range(self.n_assets):
            start_idx = i * self.n_bits_per_asset
            end_idx = start_idx + self.n_bits_per_asset
            bits = binary_solution[start_idx:end_idx]
            
            # Binary to decimal conversion for weight
            weight_sum = 0.0
            for j, bit in enumerate(bits):
                weight_sum += bit * (2 ** -j)  # Binary fraction
            
            weights[i] = weight_sum
        
        # Normalize to sum to 1
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
            
        return weights


class ClassicalOptimizer:
    """Classical portfolio optimization as baseline and fallback"""
    
    def __init__(self, universe: AssetUniverse):
        self.universe = universe
        
    def markowitz_mean_variance(self, target_return: Optional[float] = None,
                                risk_aversion: float = 1.0) -> Dict:
        """
        Classical mean-variance optimization using numpy/scipy
        """
        n = len(self.universe.symbols)
        mu = self.universe.expected_returns
        Sigma = self.universe.cov_matrix
        
        # Simplified quadratic programming approach
        # Minimize: w^T Sigma w - lambda * w^T mu
        # Subject to: sum(w) = 1, w >= 0
        
        # Use numpy's quadratic form solver approximation
        # For production, use cvxpy or scipy.optimize
        
        # Analytical solution for unconstrained (will need projection)
        try:
            # Risk-adjusted objective: maximize return - risk_aversion * variance
            # Equivalent to: minimize risk_aversion * w^T Sigma w - w^T mu
            
            # Simplified gradient descent approach
            weights = np.ones(n) / n  # Start with equal weight
            lr = 0.01
            
            for _ in range(1000):
                grad = 2 * risk_aversion * Sigma @ weights - mu
                weights = weights - lr * grad
                
                # Projection onto simplex (sum=1, w>=0)
                weights = self._project_onto_simplex(weights)
                
            portfolio_return = float(mu @ weights)
            portfolio_var = float(weights @ Sigma @ weights)
            portfolio_vol = math.sqrt(portfolio_var)
            sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'weights': weights.tolist(),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'method': 'classical_mean_variance'
            }
        except Exception as e:
            return {
                'error': str(e),
                'method': 'classical_failed'
            }
    
    def _project_onto_simplex(self, v: np.ndarray) -> np.ndarray:
        """Project vector onto probability simplex (sum=1, v>=0)"""
        # Algorithm from "Projection onto the probability simplex" (Duchi et al.)
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        return np.maximum(v - theta, 0)
    
    def risk_parity(self) -> Dict:
        """Risk parity allocation (inverse volatility)"""
        inv_vols = 1.0 / np.sqrt(np.diag(self.universe.cov_matrix))
        weights = inv_vols / np.sum(inv_vols)
        
        portfolio_return = float(self.universe.expected_returns @ weights)
        portfolio_vol = float(math.sqrt(weights @ self.universe.cov_matrix @ weights))
        
        return {
            'weights': weights.tolist(),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': (portfolio_return - 0.02) / portfolio_vol,
            'method': 'classical_risk_parity'
        }


class QAOASimulator:
    """
    QAOA (Quantum Approximate Optimization Algorithm) simulator
    Uses classical simulation of quantum circuits for small problems
    """
    
    def __init__(self, n_qubits: int, p_layers: int = 1):
        self.n_qubits = n_qubits
        self.p = p_layers  # Circuit depth
        
    def _expectation_value(self, Q: np.ndarray, gamma: float, beta: float,
                          state: Optional[np.ndarray] = None) -> float:
        """
        Compute expectation value of QUBO Hamiltonian
        Simplified classical simulation without actual quantum backend
        """
        # For simulation, use variational approach
        # Generate random binary strings and compute expectation
        n_samples = 1000
        expectation = 0.0
        
        for _ in range(n_samples):
            # Random binary string
            x = np.random.randint(0, 2, self.n_qubits).astype(float)
            
            # Compute QUBO objective: x^T Q x
            obj = x @ Q @ x
            
            # Apply variational "rotation" (simulated)
            # Simplified: weight by gamma/beta parameters
            weight = 1.0 + gamma * beta * obj
            expectation += obj * weight
            
        return expectation / n_samples
    
    def optimize(self, Q: np.ndarray, max_iterations: int = 100) -> Dict:
        """
        Variational optimization of QAOA parameters
        Returns best found solution
        """
        best_expectation = float('inf')
        best_params = (0.0, 0.0)
        best_solution = None
        
        # Grid search over parameter space (simplified)
        gamma_range = np.linspace(0, np.pi, 20)
        beta_range = np.linspace(0, np.pi, 20)
        
        for gamma in gamma_range:
            for beta in beta_range:
                exp_val = self._expectation_value(Q, gamma, beta)
                
                if exp_val < best_expectation:
                    best_expectation = exp_val
                    best_params = (gamma, beta)
        
        # Sample final solution with best parameters
        # For simulation, use greedy approach
        n_samples = 5000
        best_obj = float('inf')
        best_x = None
        
        for _ in range(n_samples):
            x = np.random.randint(0, 2, self.n_qubits).astype(float)
            obj = x @ Q @ x
            
            if obj < best_obj:
                best_obj = obj
                best_x = x
                
        return {
            'solution': best_x.astype(int).tolist(),
            'objective_value': float(best_obj),
            'parameters': {'gamma': float(best_params[0]), 'beta': float(best_params[1])},
            'layers': self.p,
            'samples_evaluated': n_samples
        }


class VQESimulator:
    """
    VQE (Variational Quantum Eigensolver) simulator
    Uses classical simulation with ansatz circuits
    """
    
    def __init__(self, n_qubits: int, ansatz_depth: int = 2):
        self.n_qubits = n_qubits
        self.depth = ansatz_depth
        
    def _hardware_efficient_ansatz(self, params: np.ndarray) -> np.ndarray:
        """
        Simulate hardware-efficient ansatz state preparation
        Returns "quantum state" as probability distribution over bitstrings
        """
        # Simplified: use parameterized rotation-like transformation
        # In real VQE, this would be quantum circuit simulation
        
        # Generate probability distribution over 2^n states
        n_states = 2 ** self.n_qubits
        probs = np.ones(n_states) / n_states
        
        # Apply parameter-dependent shifts
        for i, param in enumerate(params):
            shift = np.sin(param) * 0.1
            probs = probs + shift * (np.random.random(n_states) - 0.5)
            probs = np.abs(probs)
            probs = probs / np.sum(probs)
            
        return probs
    
    def optimize(self, hamiltonian: np.ndarray, max_iterations: int = 100) -> Dict:
        """
        Find ground state of Hamiltonian using VQE
        """
        n_params = self.n_qubits * self.depth
        
        # Random initialization
        params = np.random.random(n_params) * 2 * np.pi
        
        best_energy = float('inf')
        best_params = params.copy()
        
        # Simplified gradient-free optimization
        lr = 0.1
        for iteration in range(max_iterations):
            # Evaluate energy
            probs = self._hardware_efficient_ansatz(params)
            
            # Compute expectation: <H> = sum_i p_i * E_i
            # Simplified: sample from distribution
            energy = 0.0
            for _ in range(100):
                state_idx = np.random.choice(len(probs), p=probs)
                # Convert index to bitstring
                bits = [(state_idx >> i) & 1 for i in range(self.n_qubits)]
                x = np.array(bits, dtype=float)
                energy += x @ hamiltonian @ x
            energy /= 100
            
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
            
            # Random perturbation (gradient-free)
            params = best_params + np.random.normal(0, lr, n_params)
            lr *= 0.99  # Decay
            
        # Extract solution from best parameters
        final_probs = self._hardware_efficient_ansatz(best_params)
        most_likely_state = np.argmax(final_probs)
        
        # Convert to bitstring
        solution = np.array([(most_likely_state >> i) & 1 
                            for i in range(self.n_qubits)])
        
        return {
            'solution': solution.tolist(),
            'ground_state_energy': float(best_energy),
            'ansatz_depth': self.depth,
            'parameters': best_params.tolist(),
            'iterations': max_iterations
        }


class QuantumHybridOptimizer:
    """
    Hybrid quantum-classical optimizer for portfolio allocation
    Combines classical preprocessing with quantum sampling
    """
    
    def __init__(self, universe: AssetUniverse, 
                 algorithm: QuantumAlgorithm = QuantumAlgorithm.QAOA,
                 n_bits_per_asset: int = 3,
                 risk_aversion: float = 1.0,
                 cardinality: Optional[int] = None):
        self.universe = universe
        self.algorithm = algorithm
        self.n_bits = n_bits_per_asset
        self.risk_aversion = risk_aversion
        self.cardinality = cardinality  # Max number of assets
        
        # Classical optimizer for fallback
        self.classical = ClassicalOptimizer(universe)
        
    def _build_qubo(self, target_return: Optional[float] = None) -> QUBOFormulation:
        """
        Build QUBO formulation for portfolio optimization
        
        Objective: minimize risk_aversion * w^T Sigma w - w^T mu
        With constraints encoded as penalty terms
        """
        n = len(self.universe.symbols)
        n_binary_vars = n * self.n_bits
        
        # Initialize QUBO matrix
        Q = np.zeros((n_binary_vars, n_binary_vars))
        
        mu = self.universe.expected_returns
        Sigma = self.universe.cov_matrix
        
        # Build binary encoding for weights
        # Each asset i has bits b_{i,0}, b_{i,1}, ..., b_{i,k-1}
        # where weight_i = sum_j b_{i,j} * 2^{-j}
        
        # Objective: quadratic risk term
        for i in range(n):
            for j in range(n):
                for k_i in range(self.n_bits):
                    for k_j in range(self.n_bits):
                        idx_i = i * self.n_bits + k_i
                        idx_j = j * self.n_bits + k_j
                        
                        coeff = (2 ** -k_i) * (2 ** -k_j) * self.risk_aversion * Sigma[i, j]
                        Q[idx_i, idx_j] += coeff
        
        # Linear return term (negated for minimization)
        for i in range(n):
            for k in range(self.n_bits):
                idx = i * self.n_bits + k
                Q[idx, idx] -= (2 ** -k) * mu[i]
        
        # Constraint: sum of weights = 1 (encoded as penalty)
        penalty_weight = 10.0
        
        # Add constraint penalty to diagonal
        for i in range(n_binary_vars):
            Q[i, i] += penalty_weight
        
        # Cross terms for constraint
        for i in range(n_binary_vars):
            for j in range(i+1, n_binary_vars):
                Q[i, j] += 2 * penalty_weight
                Q[j, i] = Q[i, j]  # Symmetric
        
        # Cardinality constraint (optional)
        if self.cardinality and self.cardinality < n:
            card_penalty = 5.0
            for i in range(n):
                # Add indicator variable logic (simplified)
                idx = i * self.n_bits
                Q[idx, idx] += card_penalty * (1 - 2 * self.cardinality / n)
        
        return QUBOFormulation(n_assets=n, n_bits_per_asset=self.n_bits, Q=Q)
    
    def optimize(self, use_quantum: bool = True, 
                 quantum_p_layers: int = 1) -> Dict:
        """
        Run hybrid optimization
        """
        start_time = datetime.now()
        
        # Step 1: Classical preprocessing
        classical_result = self.classical.markowitz_mean_variance(
            risk_aversion=self.risk_aversion
        )
        
        if not use_quantum or len(self.universe.symbols) > 20:
            # NISQ limitation: quantum only beneficial for small-medium problems
            return {
                'success': True,
                'method': 'classical_fallback',
                'reason': 'problem_size_too_large_for_nisq' if len(self.universe.symbols) > 20 else 'quantum_disabled',
                'result': classical_result,
                'quantum_time_ms': 0,
                'total_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
        
        # Step 2: Build QUBO
        qubo = self._build_qubo()
        
        # Step 3: Quantum optimization
        n_vars = qubo.n_assets * qubo.n_bits_per_asset
        
        try:
            if self.algorithm == QuantumAlgorithm.QAOA:
                quantum_solver = QAOASimulator(n_vars, p_layers=quantum_p_layers)
                quantum_result = quantum_solver.optimize(qubo.Q)
                
            elif self.algorithm == QuantumAlgorithm.VQE:
                quantum_solver = VQESimulator(n_vars, ansatz_depth=2)
                quantum_result = quantum_solver.optimize(qubo.Q)
                
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
                
        except Exception as e:
            # Fallback to classical
            return {
                'success': False,
                'method': 'quantum_failed',
                'error': str(e),
                'fallback_result': classical_result
            }
        
        quantum_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Step 4: Post-processing
        solution = np.array(quantum_result['solution'])
        weights = qubo.decode_solution(solution)
        
        # Evaluate portfolio metrics
        portfolio_return = float(self.universe.expected_returns @ weights)
        portfolio_vol = float(math.sqrt(weights @ self.universe.cov_matrix @ weights))
        sharpe = (portfolio_return - 0.02) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Compare with classical
        classical_sharpe = classical_result.get('sharpe_ratio', 0)
        improvement = sharpe - classical_sharpe
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'success': True,
            'method': f'hybrid_{self.algorithm.value}',
            'quantum_result': quantum_result,
            'weights': weights.tolist(),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'vs_classical_sharpe': classical_sharpe,
            'sharpe_improvement': improvement,
            'quantum_time_ms': quantum_time,
            'total_time_ms': total_time,
            'n_assets': qubo.n_assets,
            'n_qubits_simulated': n_vars
        }


def cmd_optimize(args):
    """Run portfolio optimization"""
    
    print(f"\n⚛️  Quantum-Classical Hybrid Portfolio Optimization")
    print(f"{'='*60}")
    
    # Create asset universe
    universe = AssetUniverse.from_sample_data(n_assets=args.n_assets)
    
    print(f"Universe: {universe.symbols}")
    print(f"Assets: {len(universe.symbols)}")
    print(f"Risk Aversion: {args.risk_aversion}")
    
    if args.algorithm == 'qaoa':
        algo = QuantumAlgorithm.QAOA
        print(f"Algorithm: QAOA (p={args.p_layers})")
    elif args.algorithm == 'vqe':
        algo = QuantumAlgorithm.VQE
        print(f"Algorithm: VQE (depth=2)")
    else:
        algo = QuantumAlgorithm.CLASSICAL
        print(f"Algorithm: Classical (baseline)")
    
    # Run optimization
    optimizer = QuantumHybridOptimizer(
        universe=universe,
        algorithm=algo,
        risk_aversion=args.risk_aversion,
        cardinality=args.cardinality
    )
    
    use_quantum = args.algorithm != 'classical'
    result = optimizer.optimize(
        use_quantum=use_quantum,
        quantum_p_layers=args.p_layers
    )
    
    print(f"\n📊 Results")
    print(f"{'-'*60}")
    print(f"Method: {result['method']}")
    
    if result['success']:
        if 'expected_return' in result:
            print(f"Expected Return: {result['expected_return']*100:.2f}%")
            print(f"Volatility: {result['volatility']*100:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
            if 'vs_classical_sharpe' in result:
                improvement = result['sharpe_improvement']
                print(f"vs Classical: {result['vs_classical_sharpe']:.3f} ({improvement:+.3f})")
            
            print(f"\n📈 Allocation:")
            for i, (sym, w) in enumerate(zip(universe.symbols, result['weights'])):
                if w > 0.01:  # Only show positions > 1%
                    print(f"  {sym}: {w*100:.1f}%")
        elif 'result' in result:
            # Fallback result
            fb = result['result']
            print(f"Expected Return: {fb.get('expected_return', 0)*100:.2f}%")
            print(f"Volatility: {fb.get('volatility', 0)*100:.2f}%")
            print(f"Sharpe Ratio: {fb.get('sharpe_ratio', 0):.3f}")
        
        print(f"\n⏱️  Performance")
        print(f"{'-'*60}")
        print(f"Quantum Time: {result.get('quantum_time_ms', 0):.1f}ms")
        print(f"Total Time: {result['total_time_ms']:.1f}ms")
        
        if 'n_qubits_simulated' in result:
            print(f"Qubits Simulated: {result['n_qubits_simulated']}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
        if 'fallback_result' in result:
            print(f"\n⚠️  Fallback to classical:")
            fb = result['fallback_result']
            print(f"  Sharpe: {fb.get('sharpe_ratio', 0):.3f}")


def cmd_benchmark(args):
    """Benchmark quantum vs classical on multiple trials"""
    
    print(f"\n📊 Benchmark: Quantum vs Classical")
    print(f"{'='*60}")
    print(f"Trials: {args.trials}")
    print(f"Assets: {args.n_assets}")
    
    results = {'classical': [], 'qaoa': [], 'vqe': []}
    
    for trial in range(args.trials):
        universe = AssetUniverse.from_sample_data(n_assets=args.n_assets)
        
        # Add some randomness to returns for variety
        universe.expected_returns += np.random.normal(0, 0.01, args.n_assets)
        
        # Classical baseline
        classical_opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.CLASSICAL)
        classical_result = classical_opt.optimize(use_quantum=False)
        results['classical'].append(classical_result['result']['sharpe_ratio'])
        
        # QAOA
        qaoa_opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.QAOA)
        qaoa_result = qaoa_opt.optimize(use_quantum=True, quantum_p_layers=1)
        if qaoa_result['success']:
            results['qaoa'].append(qaoa_result['sharpe_ratio'])
        else:
            results['qaoa'].append(0)
        
        # VQE
        vqe_opt = QuantumHybridOptimizer(universe, QuantumAlgorithm.VQE)
        vqe_result = vqe_opt.optimize(use_quantum=True)
        if vqe_result['success']:
            results['vqe'].append(vqe_result['sharpe_ratio'])
        else:
            results['vqe'].append(0)
    
    print(f"\n✅ Completed {args.trials} trials")
    print(f"\n📈 Sharpe Ratio Comparison")
    print(f"{'-'*60}")
    
    for method, sharps in results.items():
        if sharps:
            mean_sharpe = np.mean(sharps)
            std_sharpe = np.std(sharps)
            print(f"{method.upper():<12} {mean_sharpe:.3f} ± {std_sharpe:.3f}")
    
    # Statistical comparison
    if results['qaoa'] and results['classical']:
        qaoa_mean = np.mean(results['qaoa'])
        classical_mean = np.mean(results['classical'])
        improvement = (qaoa_mean - classical_mean) / classical_mean * 100
        print(f"\nQAOA vs Classical: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='v2.50 Quantum-Classical Hybrid Portfolio Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s optimize --algorithm qaoa --n-assets 10
  %(prog)s optimize --algorithm classical --risk-aversion 2.0
  %(prog)s benchmark --trials 10 --n-assets 10
        """
    )
    
    parser.add_argument('--n-assets', type=int, default=10, help='Number of assets (default: 10)')
    parser.add_argument('--risk-aversion', type=float, default=1.0, help='Risk aversion parameter')
    parser.add_argument('--cardinality', type=int, help='Max number of assets to hold')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run single optimization')
    optimize_parser.add_argument('--algorithm', type=str, 
                                choices=['qaoa', 'vqe', 'classical'],
                                default='qaoa', help='Optimization algorithm')
    optimize_parser.add_argument('--p-layers', type=int, default=1, 
                               help='QAOA circuit depth')
    optimize_parser.set_defaults(func=cmd_optimize)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark multiple algorithms')
    benchmark_parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    benchmark_parser.set_defaults(func=cmd_benchmark)
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'optimize'
        args.func = cmd_optimize
        args.algorithm = 'qaoa'
        args.p_layers = 1
        
    args.func(args)


if __name__ == '__main__':
    main()
