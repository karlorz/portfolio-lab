# Portfolio-Lab Makefile — runner-agnostic ops layer
# Works with both Hermes cron AND Claude Code cron as backends.
#
# Usage:
#   make data         Fetch Yahoo Finance data
#   make dashboard    Regenerate dashboard JSON
#   make health       Generate system health.json
#   make rebalance-health Generate rebalance_health.json
#   make eval         Run strategy evaluator
#   make research     Run research agent
#   make wiki-sync    Sync findings to wiki vault
#   make build        TypeScript check + Vite production build
#   make perf         Run opt-in critical path performance budgets
#   make labs-validate  Validate existing Labs artifacts offline
#   make labs-smoke     Run Labs artifact generation smoke tests
#   make sync              Broker position sync
#   make overlay-signals    Generate all overlay signals
#   make overlay-dashboard  Generate overlay dashboard data
#   make unified-dashboard  Generate unified system dashboard
#   make garch-risk       Compute GARCH-CVaR risk metrics
#   make daily-pnl        Capture daily P&L snapshot
#   make all               Run all maintenance tasks sequentially
#   make cron-reset        Reset cron status file

SHELL := /bin/bash
PROJECT_DIR := $(shell pwd)
PORTFOLIO_LAB_PROJECT_DIR ?= $(PROJECT_DIR)
DATA_DIR := $(PROJECT_DIR)/data
CRON_UPDATE := $(PROJECT_DIR)/scripts/cron_update.py
PYTHON_RUNTIME := $(PROJECT_DIR)/scripts/python_runtime.sh
PYTHONPATH := $(PROJECT_DIR)/src:$(PYTHONPATH)
export PORTFOLIO_LAB_PROJECT_DIR
export PYTHONPATH

PERF_OUTPUT ?= $(DATA_DIR)/perf/critical_paths_latest.json
PERF_BASELINE ?= $(DATA_DIR)/perf/critical_paths_baseline.json
PERF_UPDATE_BASELINE ?= 0

# ── Help ─────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Portfolio-Lab Makefile"
	@echo ""
	@echo "  make test         Run test suite (safe: ML disabled, 3GB memory cap)"
	@echo "  make test-ml-extract  Run extracted ML-kernel tests (safe: ML disabled)"
	@echo "  make test-ml      Run full test suite including ML (requires torch/sklearn)"
	@echo "  make test-isolation  Run top-20 failing files individually (bypasses pollution)"
	@echo "  make data         Fetch Yahoo Finance market data"
	@echo "  make dashboard    Regenerate dashboard JSON files"
	@echo "  make health       Generate public/data/health.json system health monitor"
	@echo "  make rebalance-health  Generate public/data/rebalance_health.json diagnostics"
	@echo "  make eval         Run strategy evaluator (paper trading)"
	@echo "  make research     Run research agent + regime analysis"
	@echo "  make wiki-sync    Sync research findings to wiki vault"
	@echo "  make build        TypeScript check + Vite production build"
	@echo "  make perf         Run opt-in critical path performance budgets"
	@echo "  PERF_UPDATE_BASELINE=1 make perf  Refresh data/perf baseline JSON"
	@echo "  make labs-validate  Validate existing Labs artifacts offline"
	@echo "  make labs-smoke     Run Labs artifact generation smoke tests"
	@echo "  make data-quality   Audit public/data/prices.json offline"
	@echo "  make sync         Broker position reconciliation"
	@echo "  make all          Run all tasks sequentially"
	@echo "  make cron-reset   Reset cron status file to defaults"
	@echo "  make unified-dashboard  Generate unified system dashboard"
	@echo "  make daily-pnl    Capture daily P&L snapshot"
	@echo "  make mark-to-market  Update portfolio with current market prices"
	@echo "  make deploy-preview DEPLOY_HOST=sg02 [DEPLOY_REMOTE_BASE=...] [DEPLOY_PREVIEW_PORT=4173] [DEPLOY_BOOTSTRAP_PREVIEW_DATA=1]"
	@echo "  make deploy-production DEPLOY_HOST=sg01 [DEPLOY_REMOTE_BASE=...] [DEPLOY_PROD_WEB_ROOT=/var/www/portfolio-lab] [DEPLOY_RELOAD_SERVICE=caddy]"
	@echo "  make deploy-lab-app [DEPLOY_LAB_ARGS='--dry-run']  Deploy lab.karldigi.dev with systemd + Caddy"

# ── Remote Deploy ─────────────────────────────────────────────────────

DEPLOY_HOST ?=
DEPLOY_REMOTE_BASE ?=
DEPLOY_PREVIEW_PORT ?=4173
DEPLOY_BOOTSTRAP_PREVIEW_DATA ?=1
DEPLOY_PROD_WEB_ROOT ?=/var/www/portfolio-lab
DEPLOY_RELOAD_SERVICE ?=
DEPLOY_HEALTH_URL ?=
DEPLOY_EXTRA_ARGS ?=
DEPLOY_LAB_ARGS ?=

.PHONY: deploy-preview
deploy-preview:
	@[ -n "$(DEPLOY_HOST)" ] || (echo "DEPLOY_HOST is required. Example: make deploy-preview DEPLOY_HOST=sg02" && exit 1)
	@ARGS="--host $(DEPLOY_HOST) --mode preview --preview-port $(DEPLOY_PREVIEW_PORT)"; \
	if [ -n "$(DEPLOY_REMOTE_BASE)" ]; then ARGS="$$ARGS --remote-base $(DEPLOY_REMOTE_BASE)"; fi; \
	if [ "$(DEPLOY_BOOTSTRAP_PREVIEW_DATA)" = "0" ]; then ARGS="$$ARGS --no-bootstrap-preview-data"; fi; \
	if [ -n "$(DEPLOY_HEALTH_URL)" ]; then ARGS="$$ARGS --health-url $(DEPLOY_HEALTH_URL)"; fi; \
	if [ -n "$(DEPLOY_EXTRA_ARGS)" ]; then ARGS="$$ARGS $(DEPLOY_EXTRA_ARGS)"; fi; \
	echo "Running: scripts/deploy-remote.sh $$ARGS"; \
	scripts/deploy-remote.sh $$ARGS

.PHONY: deploy-production
deploy-production:
	@[ -n "$(DEPLOY_HOST)" ] || (echo "DEPLOY_HOST is required. Example: make deploy-production DEPLOY_HOST=sg01" && exit 1)
	@ARGS="--host $(DEPLOY_HOST) --mode production --prod-web-root $(DEPLOY_PROD_WEB_ROOT)"; \
	if [ -n "$(DEPLOY_REMOTE_BASE)" ]; then ARGS="$$ARGS --remote-base $(DEPLOY_REMOTE_BASE)"; fi; \
	if [ -n "$(DEPLOY_RELOAD_SERVICE)" ]; then ARGS="$$ARGS --reload-service $(DEPLOY_RELOAD_SERVICE)"; fi; \
	if [ -n "$(DEPLOY_HEALTH_URL)" ]; then ARGS="$$ARGS --health-url $(DEPLOY_HEALTH_URL)"; fi; \
	if [ -n "$(DEPLOY_EXTRA_ARGS)" ]; then ARGS="$$ARGS $(DEPLOY_EXTRA_ARGS)"; fi; \
	echo "Running: scripts/deploy-remote.sh $$ARGS"; \
	scripts/deploy-remote.sh $$ARGS

.PHONY: deploy-lab-app
deploy-lab-app:
	@scripts/deploy-lab-app.sh $(DEPLOY_LAB_ARGS)

# ── Test Suite ────────────────────────────────────────────────────────

.PHONY: perf
perf:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Critical Path Benchmarks: $$(date) ==="; \
	ARGS="--output $(PERF_OUTPUT) --baseline $(PERF_BASELINE)"; \
	if [ "$(PERF_UPDATE_BASELINE)" = "1" ]; then \
		ARGS="$$ARGS --update-baseline"; \
	else \
		ARGS="$$ARGS --fail-on-regression"; \
	fi; \
	PORTFOLIO_LAB_ENABLE_ML=0 $(PYTHON_RUNTIME) scripts/benchmark_critical_paths.py $$ARGS

.PHONY: labs-validate
labs-validate:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Labs Artifact Validation: $$(date) ==="; \
	PORTFOLIO_LAB_ENABLE_ML=0 $(PYTHON_RUNTIME) -m src.research.experiment_artifact_validator --discover-defaults

.PHONY: labs-smoke
labs-smoke:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Labs Artifact Generation Smoke: $$(date) ==="; \
	PORTFOLIO_LAB_ENABLE_ML=0 $(PYTHON_RUNTIME) -m pytest tests/test_labs_artifact_generation_smoke.py tests/test_public_data_index.py -q

.PHONY: test
test:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (safe mode): $$(date) ==="; \
	echo "  ML: disabled (PORTFOLIO_LAB_ENABLE_ML=0)"; \
	echo "  Memory cap: 3GB virtual (ulimit -v 3145728)"; \
	echo "  Heavy tests: excluded via collect_ignore"; \
	echo "  Timeout: 1200s (increased from 600s to prevent false failures)"; \
	START=$$(date +%s); \
	bash -c 'ulimit -v 3145728; \
		timeout 1200 uv run pytest tests/ -q --tb=short -p no:cacheprovider'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite: exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 124 ]; then \
		echo "TIMEOUT (124): Test suite exceeded 1200s limit. Check for hanging tests."; \
	elif [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): memory limit exceeded. Check for ML import leaks."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some tests failed (exit $$EXIT). Review output above."; \
	fi; \
	exit $$EXIT

.PHONY: test test-ml test-fast test-ml-extract

test-fast:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (fast mode): $$(date) ==="; \
	echo "  ML: disabled (PORTFOLIO_LAB_ENABLE_ML=0)"; \
	echo "  Focus: Core signal and ensemble tests only"; \
	echo "  Target: <2 minutes execution time"; \
	START=$$(date +%s); \
	PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/test_adaptive_sizing.py tests/test_adaptive_consensus.py tests/test_adaptive_ensemble_weights.py tests/test_regime_conditional_weights.py tests/test_ensemble_voter.py tests/test_regime_gate.py tests/test_ensemble_diversity_floor.py tests/test_ensemble_correlation.py tests/test_ensemble_n_eff.py tests/test_regime_bandit_integration.py -q --tb=short -p no:cacheprovider; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (fast): exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 124 ]; then \
		echo "TIMEOUT (124): Fast test suite exceeded 120s limit."; \
	elif [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): memory limit exceeded."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some tests failed (exit $$EXIT). Review output above."; \
	fi; \
	exit $$EXIT

test-ml-extract:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	PORTFOLIO_LAB_ENABLE_ML=0 ./scripts/run-tests-safe --ml-extract

# ── Test Isolation (bypasses pollution) ────────────────

.PHONY: test-isolation
test-isolation:
	@echo "=== Test Isolation Mode ===\n  Runs top-failing files individually to bypass test pollution.\n  Each file runs in a fresh process, so global-state leakage (DB,\n  singletons, module-level mocks) between unrelated test suites\n  is isolated.\n"
	@total=0; passed=0; failed=0; \
	ISOLATION_FILES="test_sentiment_client.py test_network_momentum_leadlag.py test_tsmom_overlay.py test_risk_parity_weight_overlay.py test_duration_yield_backtest.py test_fed_policy_overlay.py test_combined_strategy.py test_sentiment_analyzer.py test_ensemble_voter.py test_multi_speed_momentum.py test_international_momentum.py test_garch_cvar.py"; \
	for f in $$ISOLATION_FILES; do \
		echo "  Running $$f..."; \
		if PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest "tests/$$f" -q --tb=line -p no:cacheprovider --no-header 2>/dev/null; then \
			echo "    ✓ $$f PASSED"; \
			passed=$$((passed + 1)); \
		else \
			echo "    ✗ $$f FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
		total=$$((total + 1)); \
	done; \
	echo ""; \
	echo "=== Isolation Suite: $$passed/$$total passed, $$failed failed ==="; \
	if [ $$failed -gt 0 ]; then exit 1; fi

test-ml:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (ML mode): $$(date) ==="; \
	echo "  ML: enabled (PORTFOLIO_LAB_ENABLE_ML=1)"; \
	echo "  Heavy tests: included"; \
	echo "  WARNING: May use >3GB memory. Run on hosts with sufficient RAM."; \
	START=$$(date +%s); \
	PORTFOLIO_LAB_ENABLE_ML=1 uv run pytest tests/ -q --tb=short --include-heavy; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (ML): exit $$EXIT, duration $${DUR}s ==="

# ── Data Pipeline ────────────────────────────────────────────────────

# Retry up to 2x on bun SIGTRAP/SIGABRT (intermittent crash in cron context)
.PHONY: data
data:
	@echo "=== Data Pipeline: $$(date) ==="; \
	START=$$(date +%s); \
	retries=0; max_retries=2; \
	while [ $$retries -le $$max_retries ]; do \
		cd $(PROJECT_DIR) && export PATH="$$HOME/.bun/bin:$$PATH" && timeout 300 bun scripts/fetch-data.ts 2>&1 | tee -a $(DATA_DIR)/cron.log; \
		EXIT=$${PIPESTATUS[0]}; \
		if [ $$EXIT -eq 0 ]; then break; fi; \
		if [ $$EXIT -eq 133 ] || [ $$EXIT -eq 134 ]; then \
			retries=$$((retries + 1)); \
			echo "bun SIGTRAP/SIGABRT (exit $$EXIT), retry $$retries/$$max_retries" >> $(DATA_DIR)/cron.log; \
			sleep 5; \
		else break; fi; \
	done; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-data $$STATUS $$DUR; \
	echo "Data pipeline done ($$STATUS, $${DUR}s)"; \
	exit $$EXIT

.PHONY: data-quality
data-quality:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Public Data Quality Audit: $$(date) ==="; \
	$(PYTHON_RUNTIME) scripts/check_public_data_quality.py --app-dir $(PROJECT_DIR) --allow-repo-public-data $(DATA_QUALITY_ARGS)

# ── Dashboard ────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	@echo "=== Dashboard Generator: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.dashboard.generator 2>&1 | tee -a $(DATA_DIR)/dashboard.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-dashboard $$STATUS $$DUR; \
	exit $$EXIT

# ── Strategy Evaluator ───────────────────────────────────────────────

.PHONY: eval
eval:
	@echo "=== Strategy Evaluator: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && ALPHALAB_MODE=$${ALPHALAB_MODE:-paper} timeout 600 $(PYTHON_RUNTIME) -m src.strategy.evaluator 2>&1 | tee -a $(DATA_DIR)/eval.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-eval $$STATUS $$DUR; \
	exit $$EXIT

# ── Research Agent ───────────────────────────────────────────────────

.PHONY: research
research:
	@echo "=== Research Agent: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 600 $(PYTHON_RUNTIME) -m src.research.agent 2>&1 | tee -a $(DATA_DIR)/research.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-research $$STATUS $$DUR; \
	exit $$EXIT

# ── Wiki Sync ────────────────────────────────────────────────────────

.PHONY: wiki-sync
wiki-sync:
	@echo "=== Wiki Sync: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 1048576 && timeout 120 $(PYTHON_RUNTIME) -m src.research.wiki_sync 2>&1 | tee -a $(DATA_DIR)/wiki_sync.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-wiki-sync $$STATUS $$DUR; \
	exit $$EXIT

# ── App Build ────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "=== App Build: $$(date) ==="; \
	START=$$(date +%s); \
	export PATH="$$HOME/.bun/bin:$$PATH"; \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 600 sh -c 'bun run tsc --noEmit 2>&1 | tee -a $(DATA_DIR)/build.log && bun run build 2>&1 | tee -a $(DATA_DIR)/build.log'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-build $$STATUS $$DUR; \
	exit $$EXIT

# ── Position Sync ────────────────────────────────────────────────────

.PHONY: sync
sync:
	@echo "=== Position Sync: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 300 $(PYTHON_RUNTIME) -m src.broker.position_sync 2>&1 | tee -a $(DATA_DIR)/position_sync.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-position-sync $$STATUS $$DUR; \
	exit $$EXIT

# ── Overlay Pipeline ──────────────────────────────────────────────────

.PHONY: overlay-signals
overlay-signals:
	@echo "=== Overlay Signals: $$(date) ==="; \
	START=$$(date +%s); \
	export PROJECT_DIR="$(PROJECT_DIR)"; \
	export DATA_DIR="$(DATA_DIR)"; \
	export PYTHON_RUNTIME="$(PYTHON_RUNTIME)"; \
	timeout 600 sh -c '\
		cd $$PROJECT_DIR && ulimit -v 3145728 && \
		"$$PYTHON_RUNTIME" -m src.signals.collar_signal --save 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.signals.calendar_seasonality --save 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.signals.crypto_momentum --save 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.signals.bond_duration_signal --save 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.regime.kurtosis_regime --save 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.monitor.rebalance_health 2>&1 | tail -1'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-signals $$STATUS $$DUR; \
	exit $$EXIT

.PHONY: overlay-dashboard
overlay-dashboard:
	@echo "=== Overlay Dashboard: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.dashboard.overlay_dashboard --save 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-dashboard $$STATUS $$DUR; \
	exit $$EXIT

# Health / cron exit contract (Nagios-style + tasker):
#   0 = ok, 124 = timeout, 137 = oom, other non-zero = error.
# Recipes record STATUS via cron_update then `exit $$EXIT` so `make health`
# (and sibling cron targets) surface non-zero to tasker — never swallow.
.PHONY: health
health:
	@echo "=== Health Monitor: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.monitor.health_check 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-health $$STATUS $$DUR; \
	exit $$EXIT

.PHONY: rebalance-health
rebalance-health:
	@echo "=== Rebalance Health Export: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.monitor.rebalance_health 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-rebalance-health $$STATUS $$DUR; \
	exit $$EXIT

# ── GARCH-CVaR Risk Metrics ────────────────────────────────────────────

.PHONY: garch-risk
garch-risk:

	@echo "=== GARCH-CVaR Risk: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/compute_garch_risk.py 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-garch-risk $$STATUS $$DUR; \
	exit $$EXIT

# ── Mark-to-Market ──────────────────────────────────────────────────

.PHONY: mark-to-market
mark-to-market:
	@echo "=== Mark-to-Market: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/mark_to_market.py 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-mark-to-market $$STATUS $$DUR; \
	exit $$EXIT

# ── Daily P&L Capture ────────────────────────────────────────────────

.PHONY: daily-pnl
daily-pnl: mark-to-market
	@echo "=== Daily P&L Capture: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/capture_daily_pnl.py 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-pnl $$STATUS $$DUR; \
	exit $$EXIT

# ── Performance Attribution ────────────────────────────────────────────

.PHONY: attribution
attribution:
	@echo "=== Performance Attribution: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) -m src.monitor.performance_attribution report --save 2>&1 | tee -a $(DATA_DIR)/attribution.log; \
	EXIT=$${PIPESTATUS[0]}; \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) -m src.strategy.adaptive_ensemble_weights update --regime normal 2>&1 | tee -a $(DATA_DIR)/adaptive_weights.log; \
	EXIT2=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ] && [ $$EXIT2 -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-attribution $$STATUS $$DUR; \
	if [ $$EXIT -ne 0 ]; then exit $$EXIT; fi; \
	exit $$EXIT2

# ── Unified Dashboard ────────────────────────────────────────────────

.PHONY: unified-dashboard
unified-dashboard:
	@echo "=== Unified Dashboard: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.monitor.unified_dashboard --save 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-unified-dashboard $$STATUS $$DUR; \
	exit $$EXIT

# ── Daily Brief ──────────────────────────────────────────────────────

.PHONY: daily-brief
daily-brief:
		@echo "[$$(date '+%Y-%m-%d %H:%M:%S')] Generating daily brief..."
		@cd $(CURDIR) && uv run python -m src.monitor.daily_brief --save
		@echo "[$$(date '+%Y-%m-%d %H:%M:%S')] Daily brief saved to data/daily_brief.json"

# ── Portfolio Query ──────────────────────────────────────────────────

.PHONY: ask
ask:
	@cd $(CURDIR) && uv run python -m src.chat.portfolio_query "$(ARGS)"

# ── Run All ──────────────────────────────────────────────────────────

.PHONY: all
all: data dashboard health eval research wiki-sync sync build overlay-signals overlay-dashboard garch-risk daily-pnl attribution unified-dashboard
	@echo "=== All tasks complete: $$(date) ==="

# ── Log Retention ─────────────────────────────────────────────────────

.PHONY: prune-logs
prune-logs:
	@echo "=== Prune tasker_logs: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/prune_logs.py --keep 20 --delete-dead-health-log 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prune-logs $$STATUS $$DUR; \
	exit $$EXIT

.PHONY: prune-logs-dry-run
prune-logs-dry-run:
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/prune_logs.py --keep 20 --delete-dead-health-log --dry-run

# ── Cron Status Management ───────────────────────────────────────────

.PHONY: cron-reset
cron-reset:
	@mkdir -p $(DATA_DIR)
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-data pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-dashboard pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-health pending 0 manual

	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-eval pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-research pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-autonomous-agent pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-wiki-sync pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-build pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-position-sync pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-signals pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-dashboard pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-garch-risk pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-pnl pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-attribution pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-unified-dashboard pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prune-logs pending 0 manual
	@echo "Cron status reset: $(CRON_STATUS)"

# ── Verification ─────────────────────────────────────────────────────

.PHONY: verify-cron-sync
verify-cron-sync:
	@echo "=== Cron Backend Sync Check ==="
	@$(PYTHON_RUNTIME) -c "from cron_compat import active_backend; print(f'Active backend: {active_backend()}')"
ifeq ($(CI),true)
	@echo ""
	@echo "Preparing synthetic cron_status.json for CI..."
	@$(MAKE) --no-print-directory cron-reset
endif
	@echo ""
	@echo "Checking Makefile target coverage vs crontab..."
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/cron_verify.py --crontab $(PROJECT_DIR)/crontab
	@echo ""
	@echo "Checking cron_status.json integrity..."
ifeq ($(CI),true)
	@cd $(PROJECT_DIR) && CRON_BACKEND=tasker $(PYTHON_RUNTIME) scripts/cron_verify.py
else
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/cron_verify.py
endif
ifeq ($(CI),true)
	@echo ""
	@echo "CI mode: skipping live Hermes/system crontab overlap check"
	@echo "CI mode: skipping host-local SkillWiki/Hermes routing contract"
else
	@echo ""
	@echo "Checking live Hermes/system crontab overlap..."
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/detect_cron_overlap.py
	@echo ""
	@echo "Checking SkillWiki/Hermes routing contract..."
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/audit_routing_contract.py
endif

.PHONY: audit-routing-contract
audit-routing-contract:
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/audit_routing_contract.py

.PHONY: fetch-trends
fetch-trends:
	@echo "=== Google Trends: $$(date) ==="
	cd $(PROJECT_DIR) && uv run python scripts/fetch_google_trends.py --days 90 2>&1 | tee -a $(DATA_DIR)/cron.log
