#!/usr/bin/env python3
"""
Portfolio-Lab v2.51: AI Controller

Main entry point for the Multi-Agent Reinforcement Learning system.
Integrates with v2.24 signal integrator to provide signal-driven
MARL-based portfolio allocations.

Usage:
    # Advisory inference mode (research-shadow, non-routed)
    python -m src.agents.ai_controller --mode infer --portfolio 46/38/16
    
    # Training mode
    python -m src.agents.ai_controller --mode train --episodes 500
    
    # Backtest mode
    python -m src.agents.ai_controller --mode backtest --start 2020-01-01

Integration:
    - Consumes: src.signals.integrator (v2.24)
    - Produces: Research-shadow advisory allocation metadata
    - Format: JSON via stdout and file output
"""

import logging
import os
import numpy as np
import json
import sqlite3
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Conditional ML import — disabled by default to prevent OOM in test suites.
_ML_ENABLED = os.environ.get("PORTFOLIO_LAB_ENABLE_ML", "0") == "1"
if _ML_ENABLED:
    import torch

from src.paths import PROJECT_ROOT, BASE_ALLOCATION, MARKET_DB, sqlite_connect
from src.data.price_cache import get_prices_df

from src.agents.agent_graph import AgentGraph
from src.agents.marl_trainer import MARLTrainer, MarketEnvironment
from src.agents.base_agent import AgentObservation
from src.backtest.metrics import save_results_json

# Try to import signal integrator
try:
    from src.signals.integrator import SignalIntegrator
    SIGNAL_INTEGRATOR_AVAILABLE = True
except ImportError:
    SIGNAL_INTEGRATOR_AVAILABLE = False


# Constants
VERSION = "2.51.0"
MARL_EXECUTION_ROLE = "research_shadow_non_routed"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "marl_checkpoints"
SYNTHETIC_PRICE_DAYS = 252 * 20


def _stable_ticker_seed(ticker: str, random_state: Optional[int] = None) -> int:
    """Return a process-stable seed for synthetic fallback data."""
    seed_material = f"{ticker}:{random_state if random_state is not None else 'default'}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2**32


def _synthetic_length(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    default: int = SYNTHETIC_PRICE_DAYS,
) -> int:
    """Estimate a deterministic synthetic window length from optional dates."""
    if start_date and end_date:
        try:
            start = np.datetime64(start_date, "D")
            end = np.datetime64(end_date, "D")
            if end >= start:
                return max(2, int(np.busday_count(start, end + np.timedelta64(1, "D"))))
        except ValueError:
            logger.warning("Invalid synthetic price date window: %s to %s", start_date, end_date)
    return default


def _generate_synthetic_price_data(
    tickers: List[str],
    length: int,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generate deterministic synthetic fallback price data."""
    prices = {}
    for ticker in tickers:
        if ticker == "CASH":
            prices[ticker] = np.ones(length, dtype=np.float64)
            continue

        rng = np.random.default_rng(_stable_ticker_seed(ticker, random_state))

        # Simulate correlated returns
        base_return = 0.0002
        volatility = 0.015

        if ticker == "GLD":
            volatility = 0.012
        elif ticker == "TLT":
            volatility = 0.018

        returns = rng.normal(base_return, volatility, length)
        prices[ticker] = 100 * np.cumprod(1 + returns)

    return prices


def _load_real_price_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load aligned real prices from the shared price cache."""
    real_symbols = [ticker for ticker in tickers if ticker != "CASH"]
    if not real_symbols:
        length = _synthetic_length(start_date, end_date)
        return {"CASH": np.ones(length, dtype=np.float64)}

    try:
        prices_df = get_prices_df(symbols=real_symbols)
    except (FileNotFoundError, ImportError, ValueError, KeyError) as exc:
        logger.warning("Real price cache unavailable for AI controller: %s", exc)
        return None

    if prices_df.empty:
        return None

    missing = [symbol for symbol in real_symbols if symbol not in prices_df.columns]
    if missing:
        logger.warning("Real price cache missing AI controller symbols: %s", missing)
        return None

    window = prices_df.loc[start_date:end_date, real_symbols] if start_date or end_date else prices_df[real_symbols]
    window = window.dropna()
    if len(window) < 2:
        return None

    loaded = {
        symbol: window[symbol].to_numpy(dtype=np.float64)
        for symbol in real_symbols
    }
    if "CASH" in tickers:
        loaded["CASH"] = np.ones(len(window), dtype=np.float64)

    return {ticker: loaded[ticker] for ticker in tickers}


def load_price_data(
    tickers: List[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load historical price data for training.

    In production, connects to live data feed.
    For training, loads from cached historical data.
    """
    if tickers is None:
        tickers = ['SPY', 'GLD', 'TLT', 'CASH']

    real_prices = _load_real_price_data(tickers, start_date=start_date, end_date=end_date)
    if real_prices is not None:
        return real_prices

    length = _synthetic_length(start_date, end_date)
    return _generate_synthetic_price_data(tickers, length=length, random_state=random_state)


def create_default_portfolio() -> Dict[str, float]:
    """Create default 46/38/16/0 allocation."""
    return {**BASE_ALLOCATION, 'CASH': 0.0}


def parse_allocation_string(alloc_str: str) -> Dict[str, float]:
    """Parse allocation string (e.g., '46/38/16/0')."""
    parts = alloc_str.split('/')
    tickers = ['SPY', 'GLD', 'TLT', 'CASH']
    
    allocation = {}
    total = 0
    
    for i, part in enumerate(parts):
        if i < len(tickers):
            weight = float(part) / 100 if float(part) > 1 else float(part)
            allocation[tickers[i]] = weight
            total += weight
    
    # Normalize
    if total > 0:
        for k in allocation:
            allocation[k] /= total
    
    return allocation


def _resolve_device(requested_device: str) -> str:
    """Resolve CLI device requests without importing torch in safe mode."""
    if not requested_device.startswith("cuda"):
        return requested_device

    if not _ML_ENABLED:
        print("CUDA requested but ML is disabled; using CPU")
        return "cpu"

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"

    return requested_device


def _execution_role_metadata() -> Dict[str, Any]:
    """Return MARL live-authority boundary metadata."""
    return {
        "execution_role": MARL_EXECUTION_ROLE,
        "routed": False,
        "routed_by": None,
        "live_authoritative": False,
    }


def _ml_disabled_infer_result(
    current_allocation: Dict[str, float],
    portfolio_value: float,
    requested_device: str,
) -> Dict[str, Any]:
    """Return deterministic fail-closed CLI infer output when ML is disabled."""
    return {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "status": "safe_mode",
        "error": "ml_disabled",
        "message": (
            "PORTFOLIO_LAB_ENABLE_ML=0 disables MARL inference; "
            "no agent graph was executed and no live order routing is authorized."
        ),
        "portfolio_value": portfolio_value,
        "current_allocation": current_allocation,
        "recommended_allocation": current_allocation,
        "should_rebalance": False,
        "confidence": 0.0,
        "requested_device": requested_device,
        **_execution_role_metadata(),
    }


class AIController:
    """
    Main AI Controller for v2.51 MARL system.
    
    Bridges:
    - Signal Integrator (v2.24) -> Agent Graph
    - Agent Graph -> Portfolio allocations
    - Training -> Inference modes
    """
    
    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: Optional[Path] = None,
        use_signal_integrator: bool = True
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.version = VERSION
        
        # Initialize agent graph
        self.graph = AgentGraph(device=device)
        self.agents = self.graph.create_default_agents(hidden_dim=128)
        
        # Load checkpoint if exists
        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            self.graph.load(checkpoint_path)
        
        # Initialize signal integrator
        self.signal_integrator = None
        if use_signal_integrator and SIGNAL_INTEGRATOR_AVAILABLE:
            try:
                self.signal_integrator = SignalIntegrator()
            except Exception as exc:
                logger.warning("Signal integrator unavailable for AI controller: %s", exc)
        
        # Trainer (initialized on demand)
        self.trainer: Optional[MARLTrainer] = None
        
        # Current state
        self.current_allocation = create_default_portfolio()
        self.last_action: Optional[Dict] = None
        self.action_history: List[Dict] = []
        self.db_path = MARKET_DB

    def _fetch_price_history(self, symbol: str, days: int = 60) -> np.ndarray:
        """Fetch recent close prices from market.db. Falls back to ones if unavailable."""
        try:
            if not self.db_path.exists():
                return np.ones(days)
            with sqlite_connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT ?",
                    (symbol, days)
                )
                rows = cursor.fetchall()
            if rows and len(rows) >= days // 2:
                prices = np.array([r[0] for r in reversed(rows)], dtype=np.float64)
                if prices[-1] > 0:
                    return prices
            return np.ones(days)
        except sqlite3.Error:
            logger.exception("Failed to fetch price history for %s", symbol)
            return np.ones(days)

    def build_observation_from_integrator(
        self,
        ticker: str = "SPY",
        portfolio_value: float = 100000.0
    ) -> Optional[AgentObservation]:
        """Build observation from signal integrator output."""
        if not self.signal_integrator:
            return None
        
        try:
            # Get composite signal
            self.signal_integrator.get_composite_signal(ticker)
            
            # Extract features from signal
            # Fetch real price history from market.db
            price_history = self._fetch_price_history(ticker, 60)
            
            # Build observation
            obs = AgentObservation(
                prices=price_history,
                returns=np.array([0.0]),
                volatility=0.15,  # From signal if available
                current_weights=self.current_allocation,
                portfolio_value=portfolio_value,
                cash_available=portfolio_value * self.current_allocation.get('CASH', 0.0)
            )
            
            return obs
        except Exception as e:
            print(f"Error building observation from integrator: {e}")
            return None
    
    def infer(
        self,
        portfolio_value: float = 100000.0,
        current_allocation: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run inference to get allocation recommendation.
        
        Returns allocation decision with full metadata.
        """
        if current_allocation:
            self.current_allocation = current_allocation
        
        # Build observation
        if self.signal_integrator:
            obs = self.build_observation_from_integrator(portfolio_value=portfolio_value)
        else:
            # Fallback: synthetic observation
            obs = AgentObservation(
                prices=np.ones(60),
                returns=np.array([0.0]),
                volatility=0.15,
                current_weights=self.current_allocation,
                portfolio_value=portfolio_value,
                cash_available=portfolio_value * self.current_allocation.get('CASH', 0.0)
            )
        
        if obs is None:
            return {"error": "Failed to build observation"}
        
        # Execute agent graph
        actions = self.graph.execute_step(obs)
        
        # Extract controller output
        controller_action = actions.get('controller')
        
        if not controller_action:
            return {"error": "Controller failed to produce output"}
        
        # Build result
        result = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "current_allocation": self.current_allocation,
            "recommended_allocation": controller_action.metadata.get('allocation', self.current_allocation),
            "should_rebalance": controller_action.metadata.get('should_rebalance', False),
            "confidence": controller_action.confidence,
            "consensus_level": controller_action.metadata.get('consensus_level', 0.0),
            "agent_weights": controller_action.metadata.get('agent_weights', [0.25]*4),
            "risk_budget_applied": controller_action.metadata.get('risk_budget_applied', 1.0),
            "hedge_level": controller_action.metadata.get('hedge_level', 0.0),
            "agent_contributions": {
                agent_id: {
                    "score": action.score,
                    "direction": action.direction,
                    "confidence": action.confidence
                }
                for agent_id, action in actions.items() if agent_id != 'controller'
            },
            **_execution_role_metadata(),
        }
        
        # Update state
        self.last_action = result
        self.action_history.append(result)
        
        # Update current allocation (for next inference)
        if controller_action.metadata.get('should_rebalance'):
            new_alloc = controller_action.metadata.get('allocation', [])
            if new_alloc and len(new_alloc) >= 3:
                self.current_allocation = {
                    'SPY': new_alloc[0],
                    'GLD': new_alloc[1],
                    'TLT': new_alloc[2],
                    'CASH': new_alloc[3] if len(new_alloc) > 3 else 0.0
                }
        
        return result
    
    def train(self, n_episodes: int = 100, save_interval: int = 50) -> Dict[str, Any]:
        """
        Train the MARL system.
        
        Uses historical price data for environment simulation.
        """
        # Load training data
        print(f"Loading training data...")
        prices = load_price_data()
        
        # Create environment
        env = MarketEnvironment(
            prices=prices,
            allocations=create_default_portfolio()
        )
        
        # Create trainer
        self.trainer = MARLTrainer(
            agent_graph=self.graph,
            env=env,
            device=self.device
        )
        
        # Training
        print(f"Training for {n_episodes} episodes...")
        results = self.trainer.train(n_episodes=n_episodes, log_interval=10)
        
        # Save
        if CHECKPOINT_DIR:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CHECKPOINT_DIR / f"checkpoint_ep{n_episodes}"
            self.trainer.save(save_path)
            print(f"Saved checkpoint to {save_path}")
        
        return results
    
    def backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_value: float = 100000.0
    ) -> Dict[str, Any]:
        """Run backtest using trained agents."""
        # Load data
        prices = load_price_data(start_date=start_date, end_date=end_date)
        
        # Create environment
        env = MarketEnvironment(
            prices=prices,
            allocations=create_default_portfolio()
        )
        
        obs = env.reset()
        portfolio_values = [initial_value]
        trades = []
        
        done = False
        step = 0
        
        while not done and step < 252 * 5:  # Max 5 years
            # Get agent decision
            actions = self.graph.execute_step(obs)
            controller_action = actions.get('controller')
            
            # Step environment
            next_obs, reward, done, info = env.step(
                controller_action.metadata if controller_action else {}
            )
            
            portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))

            if info.get('turnover', 0) > 0.01:
                trades.append({
                    'step': step,
                    'turnover': info['turnover'],
                    'value': info.get('portfolio_value', portfolio_values[-1])
                })
            
            obs = next_obs
            step += 1
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        if len(returns) < 2:
            return {"version": self.version, "start_value": initial_value,
                    "end_value": portfolio_values[-1] if portfolio_values else initial_value,
                    "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0,
                    "max_drawdown": 0.0, "trades": len(trades)}

        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (np.mean(returns) * 252) / (volatility + 1e-8)
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        max_dd = np.max(drawdowns)
        
        results = {
            "version": self.version,
            "start_value": initial_value,
            "end_value": portfolio_values[-1],
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": cagr / max_dd if max_dd > 0 else 0,
            "trades": len(trades),
            "steps": step,
            "final_allocation": env.allocation
        }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "version": self.version,
            "device": str(self.device),
            "agents_loaded": list(self.graph.agents),
            "signal_integrator_connected": self.signal_integrator is not None,
            "checkpoint_loaded": self.checkpoint_path is not None and self.checkpoint_path.exists(),
            "inference_count": len(self.action_history),
            "current_allocation": self.current_allocation,
            "graph_metrics": self.graph.metrics if hasattr(self.graph, 'metrics') else {},
            **_execution_role_metadata(),
        }


def main():
    parser = argparse.ArgumentParser(description=f"Portfolio-Lab v{VERSION} AI Controller")
    parser.add_argument('--mode', choices=['infer', 'train', 'backtest', 'status'],
                       default='status', help='Operation mode')
    parser.add_argument('--portfolio', type=str, default='46/38/16/0',
                       help='Current allocation (e.g., 46/38/16/0)')
    parser.add_argument('--value', type=float, default=100000.0,
                       help='Portfolio value')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Training episodes')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path to load')
    parser.add_argument('--start', type=str, default=None,
                       help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Device
    if args.mode == 'infer' and not _ML_ENABLED:
        allocation = parse_allocation_string(args.portfolio)
        result = _ml_disabled_infer_result(
            current_allocation=allocation,
            portfolio_value=args.value,
            requested_device=args.device,
        )
        print(json.dumps(result, indent=2))
        if args.output:
            save_results_json(result, output_path=args.output)
            print(f"\nSaved to {args.output}")
        return

    device = _resolve_device(args.device)
    
    # Checkpoint
    checkpoint = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    
    # Initialize controller
    controller = AIController(
        device=device,
        checkpoint_path=checkpoint,
        use_signal_integrator=SIGNAL_INTEGRATOR_AVAILABLE
    )
    
    # Execute mode
    if args.mode == 'status':
        result = controller.get_status()
        
    elif args.mode == 'infer':
        allocation = parse_allocation_string(args.portfolio)
        result = controller.infer(
            portfolio_value=args.value,
            current_allocation=allocation
        )
        
    elif args.mode == 'train':
        result = controller.train(n_episodes=args.episodes)
        
    elif args.mode == 'backtest':
        result = controller.backtest(
            start_date=args.start,
            end_date=args.end,
            initial_value=args.value
        )
    
    # Output
    print(json.dumps(result, indent=2))
    
    if args.output:
        save_results_json(result, output_path=args.output)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
