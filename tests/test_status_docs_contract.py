"""Contract tests for slim agent instruction docs (CLAUDE.md / AGENTS.md).

Full research status lives in the SkillWiki vault; these docs must stay short
and still disclose live-execution authority, champion baseline, and MARL role.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _agent_docs() -> list[Path]:
    docs: list[Path] = []
    for name in ("CLAUDE.md", "AGENTS.md"):
        path = REPO_ROOT / name
        # AGENTS.md may be a symlink to CLAUDE.md — still resolve and read once.
        if path.exists():
            docs.append(path)
    return docs


def test_agent_docs_stay_slim_and_point_at_wiki_index():
    for path in _agent_docs():
        text = path.read_text()
        assert "## Agent instructions" in text
        assert "Do not re-expand status dumps" in text or "Do not re-expand" in text
        assert "projects/portfolio-lab/knowledge.md" in text
        assert "compound/claude-md-agent-reference.md" in text
        # Must not re-grow the old multi-section status chronicle.
        assert "## Grid Search Results" not in text
        assert "### Configured Ensemble Sources (9)" not in text
        assert "### MARL Live Placement" not in text


def test_champion_and_live_execution_semantics():
    for path in _agent_docs():
        text = path.read_text()
        assert "base-grid Sharpe 0.79" in text or "base-grid benchmark Sharpe 0.79" in text
        assert "0.95" in text  # overlay research Sharpe
        assert "signals.json.target_allocations" in text
        assert "src.broker.order_router" in text
        assert "46/38/16" in text


def test_marl_and_advisory_surfaces_not_live_authority():
    for path in _agent_docs():
        text = path.read_text()
        assert "live_authoritative: false" in text or "live_authoritative" in text
        assert "advisory" in text.lower() or "research" in text.lower()
        # Ensemble/overlays/MARL must not be described as live-routed authority.
        assert "order_router" in text
