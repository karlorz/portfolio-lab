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
#   make sync         Broker position sync
#   make all          Run all maintenance tasks sequentially
#   make cron-reset   Reset cron status file

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

# ── Run All ──────────────────────────────────────────────────────────

.PHONY: all
all: data dashboard health eval research wiki-sync sync build
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
	@echo "Cron status reset: $(CRON_STATUS)"
