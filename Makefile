# Portfolio-Lab Makefile — runner-agnostic ops layer
# Works with both Hermes cron AND Claude Code cron as backends.
#
# Usage:
#   make data         Fetch Yahoo Finance data
#   make dashboard    Regenerate dashboard JSON
#   make health       Run system health checks
#   make eval         Run strategy evaluator
#   make research     Run research agent
#   make wiki-sync    Sync findings to wiki vault
#   make build        TypeScript check + Vite production build
#   make sync              Broker position sync
#   make overlay-signals    Generate all overlay signals
#   make overlay-dashboard  Generate overlay dashboard data
#   make all               Run all maintenance tasks sequentially
#   make cron-reset        Reset cron status file

SHELL := /bin/bash
PROJECT_DIR := $(shell pwd)
DATA_DIR := $(PROJECT_DIR)/data
CRON_UPDATE := $(PROJECT_DIR)/scripts/cron_update.py
PYTHONPATH := $(PROJECT_DIR)/src:$(PYTHONPATH)
export PYTHONPATH

# ── Help ─────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo "Portfolio-Lab Makefile"
	@echo ""
	@echo "  make test         Run test suite (safe: ML disabled, 1GB memory cap)"
	@echo "  make test-ml      Run full test suite including ML (requires torch/sklearn)"
	@echo "  make data         Fetch Yahoo Finance market data"
	@echo "  make dashboard    Regenerate dashboard JSON files"
	@echo "  make health       Run system health monitor"
	@echo "  make eval         Run strategy evaluator (paper trading)"
	@echo "  make research     Run research agent + regime analysis"
	@echo "  make wiki-sync    Sync research findings to wiki vault"
	@echo "  make build        TypeScript check + Vite production build"
	@echo "  make sync         Broker position reconciliation"
	@echo "  make all          Run all tasks sequentially"
	@echo "  make cron-reset   Reset cron status file to defaults"

# ── Test Suite ────────────────────────────────────────────────────────

.PHONY: test
test:
	@echo "=== Test Suite (safe mode): $$(date) ==="; \
	echo "  ML: disabled (PORTFOLIO_LAB_ENABLE_ML=0)"; \
	echo "  Memory cap: 1GB virtual (ulimit -v)"; \
	echo "  Heavy tests: excluded via collect_ignore"; \
	START=$$(date +%s); \
	bash -c 'ulimit -v 3145728; \
		PORTFOLIO_LAB_ENABLE_ML=0 uv run pytest tests/ -q --tb=short -p no:cacheprovider'; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	echo ""; \
	echo "=== Test Suite: exit $$EXIT, duration $${DUR}s ==="; \
	if [ $$EXIT -eq 137 ]; then \
		echo "SIGKILL (137): memory limit exceeded. Check for ML import leaks."; \
	elif [ $$EXIT -ne 0 ]; then \
		echo "Some tests failed (exit $$EXIT). Review output above."; \
	fi; \
	exit $$EXIT

.PHONY: test-ml
test-ml:
	@echo "=== Test Suite (ML mode): $$(date) ==="; \
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

.PHONY: data
data:
	@echo "=== Data Pipeline: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.data.pipeline 2>&1 | tee -a $(DATA_DIR)/cron.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-data $$STATUS $$DUR; \
	echo "Data pipeline done ($$STATUS, $${DUR}s)"

# ── Dashboard ────────────────────────────────────────────────────────

.PHONY: dashboard
dashboard:
	@echo "=== Dashboard Generator: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.dashboard.generator 2>&1 | tee -a $(DATA_DIR)/dashboard.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-dashboard $$STATUS $$DUR

# ── Health Monitor ───────────────────────────────────────────────────

.PHONY: health
health:
	@echo "=== Health Monitor: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.monitor.health 2>&1 | tee -a $(DATA_DIR)/health.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-health $$STATUS $$DUR

# ── Strategy Evaluator ───────────────────────────────────────────────

.PHONY: eval
eval:
	@echo "=== Strategy Evaluator: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && ALPHALAB_MODE=$${ALPHALAB_MODE:-paper} python3 -m src.strategy.evaluator 2>&1 | tee -a $(DATA_DIR)/eval.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-eval $$STATUS $$DUR

# ── Research Agent ───────────────────────────────────────────────────

.PHONY: research
research:
	@echo "=== Research Agent: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.research.agent 2>&1 | tee -a $(DATA_DIR)/research.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-research $$STATUS $$DUR

# ── Wiki Sync ────────────────────────────────────────────────────────

.PHONY: wiki-sync
wiki-sync:
	@echo "=== Wiki Sync: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.research.wiki_sync 2>&1 | tee -a $(DATA_DIR)/wiki_sync.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-wiki-sync $$STATUS $$DUR

# ── App Build ────────────────────────────────────────────────────────

.PHONY: build
build:
	@echo "=== App Build: $$(date) ==="; \
	START=$$(date +%s); \
	export PATH="$$HOME/.bun/bin:$$PATH"; \
	cd $(PROJECT_DIR) && bun run tsc --noEmit 2>&1 | tee -a $(DATA_DIR)/build.log && bun run build 2>&1 | tee -a $(DATA_DIR)/build.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-build $$STATUS $$DUR

# ── Position Sync ────────────────────────────────────────────────────

.PHONY: sync
sync:
	@echo "=== Position Sync: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.broker.position_sync 2>&1 | tee -a $(DATA_DIR)/position_sync.log; \
	EXIT=$${PIPESTATUS[0]}; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-position-sync $$STATUS $$DUR

# ── Overlay Pipeline ──────────────────────────────────────────────────

.PHONY: overlay-signals
overlay-signals:
	@echo "=== Overlay Signals: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.signals.collar_signal --save 2>&1 | tail -1; \
	cd $(PROJECT_DIR) && python3 -m src.signals.calendar_seasonality --save 2>&1 | tail -1; \
	cd $(PROJECT_DIR) && python3 -m src.signals.crypto_momentum --save 2>&1 | tail -1; \
	cd $(PROJECT_DIR) && python3 -m src.signals.bond_duration_signal --save 2>&1 | tail -1; \
	cd $(PROJECT_DIR) && python3 -m src.regime.kurtosis_regime --save 2>&1 | tail -1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-overlay-signals $$STATUS $$DUR

.PHONY: overlay-dashboard
overlay-dashboard:
	@echo "=== Overlay Dashboard: $$(date) ==="; \
	START=$$(date +%s); \
	cd $(PROJECT_DIR) && python3 -m src.dashboard.overlay_dashboard --save 2>&1; \
	EXIT=$$?; \
	END=$$(date +%s); \
	DUR=$$((END - START)); \
	if [ $$EXIT -eq 0 ]; then STATUS="ok"; else STATUS="error"; fi; \
	python3 $(CRON_UPDATE) portfolio-lab-overlay-dashboard $$STATUS $$DUR

# ── Run All ──────────────────────────────────────────────────────────

.PHONY: all
all: data dashboard health eval research wiki-sync sync build overlay-signals overlay-dashboard
	@echo "=== All tasks complete: $$(date) ==="

# ── Cron Status Management ───────────────────────────────────────────

.PHONY: cron-reset
cron-reset:
	@mkdir -p $(DATA_DIR)
	@python3 $(CRON_UPDATE) portfolio-lab-data pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-dashboard pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-health pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-eval pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-research pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-wiki-sync pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-build pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-position-sync pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-overlay-signals pending 0 manual
	@python3 $(CRON_UPDATE) portfolio-lab-overlay-dashboard pending 0 manual
	@echo "Cron status reset: $(CRON_STATUS)"

# ── Verification ─────────────────────────────────────────────────────

.PHONY: verify-cron-sync
verify-cron-sync:
	@echo "=== Cron Backend Sync Check ==="
	@python3 -c "from cron_compat import active_backend; print(f'Active backend: {active_backend()}')"
	@echo ""
	@echo "Checking Makefile target coverage vs crontab..."
	@MISSING=0; \
	TARGETS="data dashboard health eval research wiki-sync build sync"; \
	for t in $$TARGETS; do \
		if grep -q "make.*$$t" $(PROJECT_DIR)/crontab 2>/dev/null; then \
			echo "  ✓ $$t (in crontab)"; \
		else \
			echo "  ✗ $$t MISSING from crontab"; \
			MISSING=$$((MISSING + 1)); \
		fi; \
	done; \
	if [ $$MISSING -eq 0 ]; then echo "OK: All targets synced"; else echo "FAIL: $$MISSING targets missing from crontab"; exit 1; fi
	@echo ""
	@echo "Checking cron_status.json integrity..."
	@cd $(PROJECT_DIR) && python3 scripts/cron_verify.py
