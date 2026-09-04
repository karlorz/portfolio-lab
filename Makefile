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
	@echo "  make test-gate    DEFAULT agent gate (= test-fast; <2m ensemble/signal)"
	@echo "  make test-fast    Ensemble/signal subset only (alias of test-gate)"
	@echo "  make lint         Ruff lint src/ tests/ scripts/ (CI parity, opt-in)"
	@echo "  make test         Full safe suite merge gate (ML off, 6GB VSZ, ~30-45m, 3600s)"
	@echo "  make test-unit    Safe suite excluding generator + *integration* (still ~15k tests)"
	@echo "  make test-generator  Only tests/test_generator.py (heavy dashboard path)"
	@echo "  make test-integration  Path-selected *integration* / e2e modules (S18)"
	@echo "  scripts/wait-test-exit.sh  Wait for make test exit stamp (max 60m; fail if dead)"
	@echo "  S18b suite cron: OPTIONAL (commented in crontab; not in CRON_TARGETS/tasker)"
	@echo "  make test-ml-extract  Run extracted ML-kernel tests (safe: ML disabled)"
	@echo "  make test-ml      Run full test suite including ML (requires torch/sklearn)"
	@echo "  make test-isolation  Run top-20 failing files individually (bypasses pollution)"
	@echo "  make data         Fetch Yahoo Finance market data"
	@echo "  make dashboard    Regenerate dashboard JSON files"
	@echo "  make health       Generate public/data/health.json system health monitor"
	@echo "  make daily-brief  Operator daily brief (tasker :25 hourly; dual-mode cron)"
	@echo "  make rebalance-health  Generate public/data/rebalance_health.json diagnostics"
	@echo "  make ops-regen    Post-merge operator refresh: dashboard + wiki-sync + health"
	@echo "  make eval         Run strategy evaluator (paper trading)"
	@echo "  make research     Run research agent + regime analysis"
	@echo "  make wiki-sync    Sync research findings to wiki vault"
	@echo "  make build        TypeScript check + Vite production build"
	@echo "  make perf         Run opt-in critical path performance budgets"
	@echo "  PERF_UPDATE_BASELINE=1 make perf  Refresh data/perf baseline JSON"
	@echo "  make labs-validate  Validate existing Labs artifacts offline"
	@echo "  make labs-smoke     Run Labs artifact generation smoke tests"
	@echo "  make data-quality   Audit public/data/prices.json offline"
	@echo "  make mirror-repo-public-data  Mirror live PUBLIC_DATA_DIR → repo public/data (H22b)"
	@echo "  make mirror-repo-public-data-lag  Exit 1 if repo public/data lags live"
	@echo "  make sync         Broker position reconciliation"
	@echo "  make s3-archive   Daily SeaweedFS S3 archive backup and retention prune"
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
	echo "  Memory cap: 6GB virtual (ulimit -v 6291456; raised after suite MemoryError cascade under 3GB)"; \
	echo "  Heavy tests: excluded via collect_ignore"; \
	echo "  PUBLIC_DATA_DIR: isolated mktemp (H16 — no live WWW dual-write)"; \
	echo "  Timeout: 3600s (raised after get_bl_views isolation; full safe suite ~45m on lab hosts)"; \
	START=$$(date +%s); \
	set +e; \
	bash -c 'ulimit -v 6291456; \
		ulimit -n 65536 2>/dev/null || true; \
		PUBLIC_TMP=$$(mktemp -d /tmp/plab-pytest-public.XXXXXX); \
		mkdir -p "$$PUBLIC_TMP/data"; \
		if [ -f public/data/prices.json ]; then cp -a public/data/prices.json "$$PUBLIC_TMP/data/"; \
		elif [ -f data/prices.json ]; then cp -a data/prices.json "$$PUBLIC_TMP/data/"; fi; \
		export PUBLIC_DATA_DIR="$$PUBLIC_TMP/data"; \
		export PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA=1; \
		timeout 3600 uv run pytest tests/ -q --tb=short -p no:cacheprovider; \
		EXIT=$$?; rm -rf "$$PUBLIC_TMP"; exit $$EXIT'; \
	EXIT=$$?; \
	set -e; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite: exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 124 ]; then \
		echo "TIMEOUT (124): Test suite exceeded 3600s limit. Check for hanging tests."; \
	elif [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): OOM killer / hard kill. Check for ML import leaks."; \
	elif [ $$EXIT -eq 139 ]; then \
		echo "SIGSEGV (139): virtual memory cap (ulimit -v / RLIMIT_AS) exceeded."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some tests failed (exit $$EXIT). Review output above."; \
	fi; \
	mkdir -p $(DATA_DIR) 2>/dev/null || true; \
	printf '%s\n' "{\"exit\":$$EXIT,\"ts\":\"$$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"memory_class\":$$([ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ] && echo true || echo false)}" \
		> $(DATA_DIR)/test_last_exit.json 2>/dev/null || true; \
	exit $$EXIT

.PHONY: test test-ml test-fast test-gate test-unit test-generator test-integration test-ml-extract test-ts test-browser

# S18 path segments (generator is ~6.6k lines; *integration* modules host-touch).
# test-unit = full safe suite minus those files (still ~15k tests — not a fast gate).
# Default agent mid-session gate is test-gate (= test-fast). Full gate remains `make test`.
TEST_GENERATOR_FILE := tests/test_generator.py
TEST_INTEGRATION_FILES := \
	tests/test_collect_signals_integration.py \
	tests/test_e2e_overlay_pipeline.py \
	tests/test_integration.py \
	tests/test_rebalancing_integration.py \
	tests/test_rebalancing_integration_cli.py \
	tests/test_regime_bandit_integration.py \
	tests/test_signal_backtest_integration.py \
	tests/test_signal_tsmom_integration.py \
	tests/test_tsmom_integration.py \
	tests/test_vix_vol_targeting_integration.py

# DEFAULT agent gate: alias of test-fast (<2m). Do not point agents at full `make test`.
test-gate: test-fast

# Lint gate: local parity with the CI ruff step (ci.yml lint step runs the same command).
.PHONY: lint
lint:
	@uv run ruff check src/ tests/ scripts/

# Canonical TS suite (runner matches ci.yml:60 `bun test tests/ts/`); explicit
# path only — a bare `bun test` at root would pick up stray non-suite files.
test-ts:
	@echo "=== Test Suite (ts): $$(date) ==="; \
	bun test tests/ts/; \
	exit $$?

# Dashboard browser presentation contract (playwright, tests/browser/). Local
# parity with CI-free suite; first-time prerequisite: install the pinned
# chromium headless shell via `bunx --bun playwright install chromium-headless-shell`.
# ALSO REQUIRED: the tasker backend for /api/* via the vite dev proxy
# (vite.config.ts -> 127.0.0.1:8000). It runs as the systemd unit
# `portfolio-lab-tasker.service` (install path: scripts/deploy-lab-app.sh);
# start/restart it with:
#   systemctl restart portfolio-lab-tasker
# The service holds a single-instance flock guard (data/tasker.lock): a
# second `uv run python -m src.tasker.service` exits 1 with "tasker
# singleton lock already held (pid N)" while the unit runs. The `--once`
# mirror-refresh helper is unguarded and safe to run alongside.
# If the suite hits tab-loading timeouts (analytics/risk panels not visible),
# the backend has degraded — restart it and re-run before debugging anything
# else (evidence: fresh backend 20/20 vs degraded 16-19/20, all timeout-flakes).
# webServer cold start: vite runs under plain `node` (node_modules/vite/bin/
# vite.js; `bunx --bun` stalled ~67% of playwright-spawned starts — Items
# 14/15). If a run still times out with zero server output, prewarm once with
#   node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 4173
# (Ctrl-C once it serves) and re-run. The config never reuses an existing
# PORT listener: a leftover `vite` process surfaces as an immediate
# EADDRINUSE — kill the stray process and re-run.
# Backtests-workspace smoke (Item 16): the playwright suite does NOT cover the
# backtests tab. Run `bun backtests-smoke.mjs` against a running dev server
# (SMOKE_BASE defaults to http://127.0.0.1:4173) for a 10-point render/run/
# chart smoke; no test-discovery markers, so bare `bun test` ignores it.
test-browser:
	@load=$$(awk '{print $$1}' /proc/loadavg); \
	awk -v l="$$load" 'BEGIN { if (l+0 > 3.0) { print "E2E: 1-min loadavg " l " > 3.0 — refusing to run browser suite (panel hydration is load-sensitive: reds >=4.4, greens <3, Items 14/15). Retry when load <= 3.0." > "/dev/stderr"; exit 1 } }' || exit 1; \
	echo "=== Test Suite (browser): $$(date) ==="; \
	bun run test:dashboard-browser; \
	exit $$?

test-fast:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (fast / test-gate mode): $$(date) ==="; \
	echo "  ML: disabled (PORTFOLIO_LAB_ENABLE_ML=0)"; \
	echo "  Focus: Core signal and ensemble tests only (NOT full unit suite)"; \
	echo "  Target: <2 minutes execution time"; \
	echo "  Full merge gate: make test (~30-45m). Wait helper: scripts/wait-test-exit.sh"; \
	START=$$(date +%s); \
	bash -c 'ulimit -n 65536 2>/dev/null || true; \
	PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/test_adaptive_sizing.py tests/test_adaptive_consensus.py tests/test_adaptive_ensemble_weights.py tests/test_regime_conditional_weights.py tests/test_ensemble_voter.py tests/test_regime_spec.py tests/test_regime_gate.py tests/test_ensemble_diversity_floor.py tests/test_ensemble_correlation.py tests/test_ensemble_n_eff.py tests/test_regime_bandit_integration.py tests/test_batch_ho_lag_dashboard_and_signals_restamp.py tests/test_batch_ih_health_ops_reconcile_timeout.py -q --tb=short -p no:cacheprovider; \
	exit $$?'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (fast/gate): exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 124 ]; then \
		echo "TIMEOUT (124): Fast test suite exceeded 120s limit."; \
	elif [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): OOM killer / hard kill."; \
	elif [ $$EXIT -eq 139 ]; then \
		echo "SIGSEGV (139): virtual memory cap (ulimit -v) exceeded."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some tests failed (exit $$EXIT). Review output above."; \
	fi; \
	exit $$EXIT

# Unit segment: safe suite excluding generator + path-selected integration modules
test-unit:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (unit segment / S18): $$(date) ==="; \
	echo "  ML: disabled; Memory: 6GB virtual; PUBLIC: isolated"; \
	echo "  Excludes: test_generator.py + *integration* path list"; \
	echo "  Timeout: 2400s (full suite uses 3600s)"; \
	START=$$(date +%s); \
	bash -c 'ulimit -v 6291456 2>/dev/null || true; \
		ulimit -n 65536 2>/dev/null || true; \
		PUBLIC_TMP=$$(mktemp -d /tmp/plab-pytest-public.XXXXXX); \
		mkdir -p "$$PUBLIC_TMP/data"; \
		export PUBLIC_DATA_DIR="$$PUBLIC_TMP/data"; \
		export PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA=1; \
		export PORTFOLIO_LAB_ENABLE_ML=0; \
		IGNORE_ARGS="--ignore=$(TEST_GENERATOR_FILE)"; \
		for f in $(TEST_INTEGRATION_FILES); do \
			IGNORE_ARGS="$$IGNORE_ARGS --ignore=$$f"; \
		done; \
		timeout 2400 uv run pytest tests/ -q --tb=short -p no:cacheprovider $$IGNORE_ARGS; \
		EXIT=$$?; rm -rf "$$PUBLIC_TMP"; exit $$EXIT'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (unit): exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 124 ]; then \
		echo "TIMEOUT (124): Unit segment exceeded 2400s."; \
	elif [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): OOM killer / hard kill (6GB virtual)."; \
	elif [ $$EXIT -eq 139 ]; then \
		echo "SIGSEGV (139): virtual memory cap (ulimit -v / RLIMIT_AS) exceeded."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some unit-segment tests failed (exit $$EXIT)."; \
	fi; \
	exit $$EXIT

# Generator-only segment (dashboard / public dual-write heavy)
test-generator:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (generator segment / S18): $$(date) ==="; \
	echo "  File: $(TEST_GENERATOR_FILE)"; \
	echo "  Timeout: 1200s"; \
	START=$$(date +%s); \
	bash -c 'ulimit -v 6291456 2>/dev/null || true; \
		PUBLIC_TMP=$$(mktemp -d /tmp/plab-pytest-public.XXXXXX); \
		mkdir -p "$$PUBLIC_TMP/data"; \
		export PUBLIC_DATA_DIR="$$PUBLIC_TMP/data"; \
		export PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA=1; \
		export PORTFOLIO_LAB_ENABLE_ML=0; \
		timeout 1200 uv run pytest $(TEST_GENERATOR_FILE) -q --tb=short -p no:cacheprovider; \
		EXIT=$$?; rm -rf "$$PUBLIC_TMP"; exit $$EXIT'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (generator): exit $$EXIT, duration $${DUR}s ==="; \
	exit $$EXIT

# Integration path segment (host-touching / multi-module flows)
test-integration:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Test Suite (integration segment / S18): $$(date) ==="; \
	echo "  Files: path-selected *integration* + e2e modules"; \
	echo "  Timeout: 1200s"; \
	START=$$(date +%s); \
	bash -c 'ulimit -v 6291456 2>/dev/null || true; \
		PUBLIC_TMP=$$(mktemp -d /tmp/plab-pytest-public.XXXXXX); \
		mkdir -p "$$PUBLIC_TMP/data"; \
		export PUBLIC_DATA_DIR="$$PUBLIC_TMP/data"; \
		export PORTFOLIO_LAB_ALLOW_REPO_PUBLIC_DATA=1; \
		export PORTFOLIO_LAB_ENABLE_ML=0; \
		timeout 1200 uv run pytest $(TEST_INTEGRATION_FILES) -q --tb=short -p no:cacheprovider; \
		EXIT=$$?; rm -rf "$$PUBLIC_TMP"; exit $$EXIT'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite (integration): exit $$EXIT, duration $${DUR}s ==="; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-data $$STATUS $$DUR; \
	$(PYTHON_RUNTIME) -c "from src.dashboard.cron_scheduler_section import refresh_public_health_cron_section; refresh_public_health_cron_section()" 2>/dev/null || true; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after data (repo public lag; non-blocking)"; \
	fi; \
	echo "Data pipeline done ($$STATUS, $${DUR}s)"; \
	exit $$EXIT

.PHONY: data-quality
data-quality:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Public Data Quality Audit: $$(date) ==="; \
	$(PYTHON_RUNTIME) scripts/check_public_data_quality.py --app-dir $(PROJECT_DIR) --allow-repo-public-data $(DATA_QUALITY_ARGS)

# Batch BW / H22b: mirror operator PUBLIC_DATA_DIR → repo public/data (stale SHA fix)
.PHONY: mirror-repo-public-data
mirror-repo-public-data:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	echo "=== Mirror live public data → repo public/data: $$(date) ==="; \
	$(PYTHON_RUNTIME) scripts/mirror_repo_public_data.py $(MIRROR_REPO_PUBLIC_ARGS)

.PHONY: mirror-repo-public-data-lag
mirror-repo-public-data-lag:
	@source scripts/test-repo-guard.sh && guard_ensure_portfolio_lab; \
	$(PYTHON_RUNTIME) scripts/mirror_repo_public_data.py --lag-only $(MIRROR_REPO_PUBLIC_ARGS)

# ── Dashboard ────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	@echo "=== Dashboard Generator: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 180 $(PYTHON_RUNTIME) -m src.dashboard.generator 2>&1 | tee -a $(DATA_DIR)/dashboard.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-dashboard $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after dashboard (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 2 ]; then STATUS="blocked"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-wiki-sync $$STATUS $$DUR; \
	exit $$EXIT

# ── Post-merge operator artifact regen ────────────────────────────────
# After kill-authority / dashboard projection / graduation / wiki-sync code
# lands, scheduled crons can lag 15–120m. Run this before treating the fix
# as live on WWW/operators:
#   make ops-regen
# Paths that need regen:
#   - src/dashboard/**, kill projection → dashboard (+ health)
#   - scripts/compute_garch_risk.py → garch-risk (public garch_cvar.json dual-write)
#   - src/research/wiki_sync.py, graduation SSOT → wiki-sync
#   - src/monitor/health_check.py → health
#
# LAST-WRITER CONTRACT (signals.json generator_git_sha):
#   Full `make dashboard` stamps generator_git_sha_status=full_generate.
#   Health kill-refresh and bounded alt-data partials intentionally clear the
#   live sha (partial_patch) and preserve last_full_generator_git_sha.
#   ops-regen therefore runs health BEFORE dashboard so the full generate is
#   the last writer of signals.json after a controlled regen. Standalone
#   health/alt-data crons may still leave partial_patch when they run later —
#   that is documented honesty, not a bug.
.PHONY: ops-regen
ops-regen:
	@echo "=== Ops regen (post-merge operator surfaces): $$(date) ==="
	@$(MAKE) --no-print-directory garch-risk
	@$(MAKE) --no-print-directory wiki-sync
	@$(MAKE) --no-print-directory health
	@# Full dashboard LAST so signals.json retains full_generate tip stamp
	@$(MAKE) --no-print-directory dashboard
	@# Batch BX: soft-gate repo public/data mirror (never block ops-regen)
	@$(MAKE) --no-print-directory mirror-repo-public-data || \
		echo "WARN: mirror-repo-public-data soft-failed (repo public lag; non-blocking)"
	@echo "=== Ops regen complete: $$(date) ==="
	@echo "Verify: PUBLIC_DATA_DIR signals.json generator_git_sha matches git rev-parse --short HEAD (full_generate last writer)"
	@echo "Verify: PUBLIC_DATA_DIR/garch_cvar.json exists; garch_active honest vs coverage_pass"
	@echo "Verify: make mirror-repo-public-data-lag → exit 0 (repo public/data vs live)"

# ── App Build ────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "=== App Build: $$(date) ==="; \
	START=$$(date +%s); \
	export PATH="$$HOME/.bun/bin:$$PATH"; \
	cd $(PROJECT_DIR) && ulimit -v 8388608 && timeout 600 bash -o pipefail -c 'bun run tsc --noEmit 2>&1 | tee -a $(DATA_DIR)/build.log && bun run build 2>&1 | tee -a $(DATA_DIR)/build.log'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
		"$$PYTHON_RUNTIME" -m src.signals.alternative_data_signal --generate 2>&1 | tail -1 && \
		"$$PYTHON_RUNTIME" -m src.monitor.rebalance_health 2>&1 | tail -1'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-signals $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after overlay-signals (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-overlay-dashboard $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after overlay-dashboard (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-health $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after health (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-rebalance-health $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after rebalance-health (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-pnl $$STATUS $$DUR; \
	exit $$EXIT

# One-shot / ops: rewrite paper history daily_return from NAV chain (Batch CG).
# Does not stamp cron_status (not a scheduled tasker job).
.PHONY: backfill-paper-returns
backfill-paper-returns:
	@echo "=== Backfill paper history returns: $$(date) ==="
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/capture_daily_pnl.py --backfill-paper-history $(if $(DRY_RUN),--dry-run,)

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
	if [ $$EXIT -eq 0 ] && [ $$EXIT2 -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after attribution (repo public lag; non-blocking)"; \
	fi; \
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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-unified-dashboard $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after unified-dashboard (repo public lag; non-blocking)"; \
	fi; \
	exit $$EXIT

# ── Daily Brief ──────────────────────────────────────────────────────

.PHONY: daily-brief
daily-brief:
	@echo "=== Daily Brief: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 120 $(PYTHON_RUNTIME) -m src.monitor.daily_brief --save 2>&1 | tee -a $(DATA_DIR)/daily_brief.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-brief $$STATUS $$DUR; \
	if [ $$EXIT -eq 0 ]; then \
	  $(MAKE) --no-print-directory mirror-repo-public-data || \
	  echo "WARN: mirror-repo-public-data soft-failed after daily-brief (repo public lag; non-blocking)"; \
	fi; \
	exit $$EXIT

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
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prune-logs $$STATUS $$DUR; \
	exit $$EXIT

.PHONY: prune-logs-dry-run
prune-logs-dry-run:
	@cd $(PROJECT_DIR) && $(PYTHON_RUNTIME) scripts/prune_logs.py --keep 20 --delete-dead-health-log --dry-run

# ── Prod ideas (ops SSOT → machine channel delta; badge-only promote) ──
# Hourly hybrid prod→dev capture. Never creates planned work items.
# ML off; machine JSON under data/prod_idea_channels.json is the SSOT.

.PHONY: prod-ideas
prod-ideas:
	@echo "=== Prod Ideas: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && \
	  export PORTFOLIO_LAB_ENABLE_ML=0; \
	  ulimit -v 3145728 && \
	  timeout 60 $(PYTHON_RUNTIME) -m src.monitor.prod_ideas 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prod-ideas $$STATUS $$DUR; \
	exit $$EXIT

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
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-mark-to-market pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-pnl pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-attribution pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-unified-dashboard pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prune-logs pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-prod-ideas pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-fetch-trends pending 0 manual
	@$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-daily-brief pending 0 manual
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
	@echo "=== Google Trends: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ulimit -v 3145728 && timeout 300 $(PYTHON_RUNTIME) scripts/fetch_google_trends.py --days 90 2>&1 | tee -a $(DATA_DIR)/cron.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; \
	elif [ $$EXIT -eq 3 ]; then STATUS="rate_limited"; \
	elif [ $$EXIT -eq 124 ]; then STATUS="timeout"; \
	elif [ $$EXIT -eq 137 ] || [ $$EXIT -eq 139 ]; then STATUS="oom"; \
	else STATUS="error"; fi; \
	$(PYTHON_RUNTIME) $(CRON_UPDATE) portfolio-lab-fetch-trends $$STATUS $$DUR; \
	exit $$EXIT

# ── SeaweedFS S3 Daily Archive (Track A) ─────────────────────────────

.PHONY: s3-archive
s3-archive:
	@echo "=== SeaweedFS S3 Daily Archive: $$(date) ==="; \
	timeout 2400 $(PROJECT_DIR)/scripts/cron/portfolio-lab-s3-archive.sh
