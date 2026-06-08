#!/usr/bin/env python3
"""Tests for src/agents/ai_controller.py — AI Controller.

Runs under PORTFOLIO_LAB_ENABLE_ML=0 (safe mode). Tests the controller
infrastructure, CLI parsing, and data utility functions. Does NOT test
ML inference/training (requires torch).
"""
import os
import subprocess
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

from src.agents.ai_controller import (
    create_default_portfolio,
    load_price_data,
    parse_allocation_string,
    AIController,
    VERSION,
    main,
)
import src.agents.ai_controller as ai_controller_module


class TestCreateDefaultPortfolio:
    def test_returns_expected_weights(self):
        pf = create_default_portfolio()
        assert pf["SPY"] == 0.46
        assert pf["GLD"] == 0.38
        assert pf["TLT"] == 0.16

    def test_weights_sum_to_one(self):
        pf = create_default_portfolio()
        assert abs(sum(pf.values()) - 1.0) < 0.001

    def test_has_four_keys_including_cash(self):
        pf = create_default_portfolio()
        assert len(pf) == 4  # SPY, GLD, TLT, CASH


class TestParseAllocationString:
    def test_standard_format(self):
        result = parse_allocation_string("46/38/16/0")
        assert result["SPY"] == 0.46
        assert result["GLD"] == 0.38
        assert result["TLT"] == 0.16
        assert result["CASH"] == 0.0

    def test_three_asset_format(self):
        result = parse_allocation_string("50/30/20")
        assert result["SPY"] == 0.50
        assert result["GLD"] == 0.30
        assert result["TLT"] == 0.20

    def test_single_value(self):
        result = parse_allocation_string("100")
        assert result["SPY"] == 1.0

    def test_zero_allocation(self):
        result = parse_allocation_string("0/0/0/100")
        assert result["SPY"] == 0.0
        assert result["CASH"] == 1.0


class TestLoadPriceData:
    def test_loads_aligned_real_prices_from_shared_cache(self, monkeypatch):
        import pandas as pd

        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        df = pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0],
                "GLD": [150.0, 151.0, 152.0],
                "TLT": [90.0, 91.0, 92.0],
            },
            index=dates,
        )
        calls = []

        def fake_get_prices_df(symbols=None):
            calls.append(symbols)
            return df.copy()

        monkeypatch.setattr(ai_controller_module, "get_prices_df", fake_get_prices_df, raising=False)

        prices = load_price_data(
            ["SPY", "GLD", "TLT", "CASH"],
            start_date="2020-01-02",
            end_date="2020-01-03",
        )

        assert calls == [["SPY", "GLD", "TLT"]]
        np.testing.assert_array_equal(prices["SPY"], np.array([101.0, 102.0]))
        np.testing.assert_array_equal(prices["GLD"], np.array([151.0, 152.0]))
        np.testing.assert_array_equal(prices["TLT"], np.array([91.0, 92.0]))
        np.testing.assert_array_equal(prices["CASH"], np.ones(2))

    def test_synthetic_fallback_is_stable_across_python_processes(self):
        script = (
            "import json;"
            "from src.agents.ai_controller import load_price_data;"
            "series = load_price_data(['__MISSING__'])['__MISSING__'][:8];"
            "print(json.dumps([round(float(x), 8) for x in series]))"
        )
        env = os.environ.copy()
        env["PORTFOLIO_LAB_ENABLE_ML"] = "0"
        env.pop("PYTHONHASHSEED", None)

        first = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        second = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert first.stdout == second.stdout


class TestAIControllerInit:
    def test_creates_controller(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            assert ctrl.device == "cpu"
            assert ctrl.current_allocation == create_default_portfolio()
            assert ctrl.trainer is None
            assert ctrl.action_history == []

    def test_default_allocation(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            assert ctrl.current_allocation["SPY"] == 0.46

    def test_no_signal_integrator_by_default(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            assert ctrl.signal_integrator is None


class TestAIControllerStatus:
    def test_get_status_structure(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.metrics = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            status = ctrl.get_status()
            assert "version" in status
            assert "device" in status
            assert "agents_loaded" in status
            assert "signal_integrator_connected" in status
            assert "inference_count" in status
            assert "current_allocation" in status

    def test_status_shows_no_integrator(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.metrics = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            status = ctrl.get_status()
            assert status["signal_integrator_connected"] is False

    def test_status_shows_zero_inferences(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.metrics = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            status = ctrl.get_status()
            assert status["inference_count"] == 0


class TestAIControllerCli:
    def test_status_cuda_request_falls_back_in_safe_mode_without_torch(self, monkeypatch, capsys):
        monkeypatch.setenv("PORTFOLIO_LAB_ENABLE_ML", "0")
        monkeypatch.setattr(
            "sys.argv",
            ["ai_controller", "--mode", "status", "--device", "cuda"],
        )

        with patch("src.agents.ai_controller.AIController") as mock_controller_cls:
            mock_controller = MagicMock()
            mock_controller.get_status.return_value = {
                "version": VERSION,
                "device": "cpu",
                "agents_loaded": [],
            }
            mock_controller_cls.return_value = mock_controller

            main()

        mock_controller_cls.assert_called_once()
        assert mock_controller_cls.call_args.kwargs["device"] == "cpu"
        output = capsys.readouterr().out
        assert "CUDA requested but ML is disabled" in output
        assert '"device": "cpu"' in output

    def test_status_cuda_request_cli_exits_cleanly_in_safe_mode(self):
        env = os.environ.copy()
        env["PORTFOLIO_LAB_ENABLE_ML"] = "0"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.agents.ai_controller",
                "--mode",
                "status",
                "--device",
                "cuda",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert "CUDA requested but ML is disabled" in result.stdout
        assert '"device": "cpu"' in result.stdout


class TestAIControllerFetchPriceHistory:
    def test_returns_fallback_when_no_db(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            ctrl.db_path = Path("/nonexistent/market.db")
            prices = ctrl._fetch_price_history("SPY", days=60)
            assert len(prices) == 60
            assert all(p == 1.0 for p in prices)

    def test_returns_requested_length(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            ctrl.db_path = Path("/nonexistent/market.db")
            prices = ctrl._fetch_price_history("SPY", days=30)
            assert len(prices) == 30


class TestAIControllerInfer:
    def test_infer_without_integrator(self):
        """Infer with synthetic observation (no signal integrator)."""
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.execute_step.return_value = {"controller": MagicMock(
                allocation_delta={"SPY": 0.01, "GLD": -0.01, "TLT": 0.0},
                confidence=0.5,
            )}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            result = ctrl.infer(portfolio_value=100000.0)
            assert isinstance(result, dict)

    def test_infer_with_custom_allocation(self):
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.execute_step.return_value = {"controller": MagicMock(
                allocation_delta={"SPY": 0.01, "GLD": -0.01, "TLT": 0.0},
                confidence=0.5,
            )}
            mock_graph_cls.return_value = mock_graph
            ctrl = AIController(use_signal_integrator=False)
            custom = {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2}
            ctrl.infer(
                portfolio_value=50000.0,
                current_allocation=custom,
            )
            assert ctrl.current_allocation == custom


class TestAIControllerTrain:
    def test_train_returns_result_dict(self):
        """Training should return a result dict."""
        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls, \
             patch("src.agents.ai_controller.MARLTrainer") as mock_trainer_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph_cls.return_value = mock_graph
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {"episodes": 1, "best_sharpe": 0.5}
            mock_trainer_cls.return_value = mock_trainer
            ctrl = AIController(use_signal_integrator=False)
            result = ctrl.train(n_episodes=1)
            assert isinstance(result, dict)


class TestAIControllerBacktestDateRange:
    def test_backtest_passes_date_range_to_price_loader(self, monkeypatch):
        captured = {}

        def fake_load_price_data(tickers=None, start_date=None, end_date=None):
            captured["tickers"] = tickers
            captured["start_date"] = start_date
            captured["end_date"] = end_date
            return {
                "SPY": np.linspace(100, 110, 20),
                "GLD": np.linspace(150, 152, 20),
                "TLT": np.linspace(90, 91, 20),
                "CASH": np.ones(20),
            }

        monkeypatch.setattr(ai_controller_module, "load_price_data", fake_load_price_data)

        with patch("src.agents.ai_controller.AgentGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.agents = {}
            mock_graph.execute_step.return_value = {
                "controller": MagicMock(metadata={})
            }
            mock_graph_cls.return_value = mock_graph

            ctrl = AIController(use_signal_integrator=False)
            result = ctrl.backtest(start_date="2020-01-02", end_date="2020-01-10")

        assert captured["start_date"] == "2020-01-02"
        assert captured["end_date"] == "2020-01-10"
        assert result["steps"] > 0


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0
