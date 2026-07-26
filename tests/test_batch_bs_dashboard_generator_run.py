"""Batch BS: DashboardGenerator.run must stay a class method (ops CLI).

Batch BP inserted ``refresh_graduation_dual_surfaces`` mid-class so methods
after it (including ``run``) nested inside the module function. Cron
``make data`` / ``make dashboard`` then failed with:
AttributeError: 'DashboardGenerator' object has no attribute 'run'
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from src.dashboard import generator as generator_mod
from src.dashboard.generator import DashboardGenerator, refresh_graduation_dual_surfaces


def test_dashboard_generator_has_run_method():
    assert hasattr(DashboardGenerator, "run")
    assert callable(DashboardGenerator.run)
    # Bound method on instance
    # Avoid full __init__ side effects when possible
    assert "run" in dir(DashboardGenerator)


def test_run_is_class_body_method_not_nested_in_refresh():
    """AST guard: ``run`` is a direct child of DashboardGenerator class body."""
    src_path = Path(inspect.getsourcefile(generator_mod))
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DashboardGenerator"
    )
    class_methods = {
        n.name
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run" in class_methods
    assert "generate_explainability_json" in class_methods
    assert "generate_risk_decomposition_json" in class_methods
    assert "generate_graduation_json" in class_methods

    # refresh_graduation_dual_surfaces is module-level, not nested
    mod_funcs = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "refresh_graduation_dual_surfaces" in mod_funcs

    # Ensure run is NOT nested inside any module-level function body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nested = {
                n.name
                for n in ast.walk(node)
                if isinstance(n, ast.FunctionDef) and n is not node
            }
            assert "run" not in nested, f"run nested under module function {node.name}"


def test_refresh_graduation_dual_surfaces_is_module_callable():
    assert callable(refresh_graduation_dual_surfaces)
    sig = inspect.signature(refresh_graduation_dual_surfaces)
    assert "public_dir" in sig.parameters
    assert "paper_trading_builder" in sig.parameters
