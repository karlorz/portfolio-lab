#!/usr/bin/env python3
"""
Print-to-Logger Migration Script — Phase 2 (v2)
Migrates all print() calls to logger.info() in target files.

This version properly:
- Adds import logging + logger at MODULE LEVEL (not inside functions)
- Handles indentation correctly
- Preserves CLI main() function prints (only migrates non-CLI files)

Usage: python3 scripts/migrate_prints_to_logger_v2.py [--run] [--revert]
  --run: Actually apply changes
  (default: dry-run, show what would change)
"""
import ast
import os
import sys
import re

TARGET_FILES = [
    # Tier 1 — No logger yet (add import logging + logger)
    'src/strategy/regime_sentiment.py',
    'src/signals/stacking_feature_engine.py',
    'src/data/alternative_data.py',
    'src/backtest/combined_strategy.py',
    'src/backtest/alt_data_walkforward_stress.py',
    'src/backtest/car25.py',
    'src/backtest/run_actual_ubt_validation.py',
    'src/agents/risk_agent_hmm.py',
    'src/llm/sentiment_client.py',
    'src/backtest/alternative_data_backfill.py',
    # Tier 2 — Has logger but still uses print()
    'src/signals/alternative_data_signal.py',
    'src/strategy/convexity_harvest.py',
    'src/strategy/factor_rotation.py',
    'src/strategy/orchestrator_ensemble_bridge.py',
    'src/regime/kurtosis_regime.py',
    'src/backtest/dbc_weight_sweep.py',
    # Extra — in spec but missing from v1 script
    'src/strategy/vol_parity_allocator.py',
]


def has_print_in_non_cli(content: str) -> bool:
    """Check if file has print() calls outside of main() CLI function."""
    # Count total prints
    total = content.count('print(')
    if total == 0:
        return False
    # If all prints are inside main() or __name__ guard, skip migration
    # (those are legitimate CLI outputs)
    lines = content.split('\n')
    in_main = False
    non_cli_prints = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def main('):
            in_main = True
        elif stripped.startswith('def ') and not stripped.startswith('def main('):
            in_main = False
        elif stripped == 'if __name__ == ' + "'__main__':" or stripped == 'if __name__ == "__main__":':
            in_main = True
        elif in_main and 'print(' in stripped:
            continue  # Skip prints in main()
        elif not in_main and 'print(' in stripped and not stripped.startswith('#'):
            non_cli_prints += 1
    return non_cli_prints > 0


def module_level_imports(content: str) -> int:
    """Find the last module-level import line number (0-indexed)."""
    lines = content.split('\n')
    last_import = -1
    indent_level = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Track indentation to detect function/class bodies
        if stripped == '' or stripped.startswith('#'):
            continue
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces == 0:
            indent_level = 0
            if stripped.startswith(('import ', 'from ')) and 'import' in stripped:
                last_import = i
        # If we hit a non-import, non-decorator, non-comment line at indent 0, stop looking
        if (leading_spaces == 0 and not stripped.startswith(('import ', 'from ', '#', '"', "'", '@', '', '__all__'))
            and 'import' not in stripped):
            pass  # Don't reset, just keep tracking
    return last_import


def add_logger_boilerplate(content: str) -> str:
    """Add import logging and logger = getLogger at module level if missing."""
    if 'logger = logging.getLogger' in content:
        return content  # Already has logger
    
    lines = content.split('\n')
    
    # Find the last module-level import
    last_import = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            continue
        leading_spaces = len(line) - len(line.lstrip())
        # Module level = 0 indentation
        if leading_spaces > 0:
            continue
        if stripped.startswith(('import ', 'from ')) or 'import ' in stripped:
            last_import = i
    
    if last_import < 0:
        return content
    
    # Find where to insert: after the last import and any blank/docstring lines
    insert_at = last_import + 1
    # Skip blank lines and module-level docstring closures
    while insert_at < len(lines) and (lines[insert_at].strip() == '' or lines[insert_at].strip() in ('"""', "'''")):
        insert_at += 1
    
    # Insert after last import module-level line
    # Logging imports
    logger_block = [
        'import logging',
        '',
        'logger = logging.getLogger(__name__)',
        '',
    ]
    
    new_lines = lines[:insert_at] + logger_block + lines[insert_at:]
    return '\n'.join(new_lines)


def replace_prints_with_logger(content: str, filepath: str) -> tuple[str, int]:
    """Replace print() with logger.info() outside main() blocks."""
    lines = content.split('\n')
    new_lines = []
    replacements = 0
    
    in_main = False
    brace_depth = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Track if we're inside main() or __main__ guard
        if stripped.startswith('def main('):
            in_main = True
        elif re.match(r'^def \w+', stripped) and not stripped.startswith('def main('):
            in_main = False
        elif stripped == "if __name__ == '__main__':" or stripped == 'if __name__ == "__main__":':
            in_main = True
        
        # Skip lines inside main() — those are legitimate CLI outputs
        if in_main and 'print(' in stripped and not stripped.startswith('#'):
            new_lines.append(line)
            continue
        
        if 'print(' in stripped and not stripped.startswith('#'):
            # Check if it's not __repr__/__str__
            if '__repr__' in stripped or '__str__' in stripped:
                new_lines.append(line)
                continue
            if 'print(' in stripped:
                # Simple replacement
                new_line = line.replace('print(', 'logger.info(')
                count = line.count('print(')
                replacements += count
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines), replacements


def migrate_file(filepath: str, dry_run: bool = True) -> int:
    """Migrate print() to logger.info() in a file. Returns replacement count."""
    full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), filepath) if not os.path.exists(filepath) else filepath
    
    if not os.path.exists(full_path):
        print(f"  MISSING: {full_path}")
        return 0
    
    with open(full_path) as f:
        content = f.read()
    
    original = content
    
    # Check if migration is needed
    total_prints = content.count('print(')
    if total_prints == 0:
        print(f"  SKIP (no prints): {filepath}")
        return 0
    
    # Step 1: Add logger boilerplate at module level
    content = add_logger_boilerplate(content)
    
    # Step 2: Replace print() with logger.info() outside main()
    content, replacements = replace_prints_with_logger(content, filepath)
    
    if content == original:
        print(f"  NO CHANGE: {filepath} ({total_prints} prints remain)")
        return 0
    
    if dry_run:
        # Show what would change
        diff_lines = []
        orig_lines = original.split('\n')
        new_lines = content.split('\n')
        for i, (o, n) in enumerate(zip(orig_lines, new_lines)):
            if o != n:
                diff_lines.append(f"  L{i+1}: -{o}")
                diff_lines.append(f"        +{n}")
        print(f"  WOULD CHANGE: {filepath} ({total_prints} prints -> logger, {replacements} replacements)")
        if diff_lines:
            for dl in diff_lines[:20]:
                print(dl)
            if len(diff_lines) > 20:
                print(f"  ... and {len(diff_lines) - 20} more changes")
        return 0
    else:
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"  MIGRATED: {filepath} ({total_prints} prints -> logger, {replacements} replacements)")
        return replacements


def verify_syntax(filepath: str) -> bool:
    """Verify a Python file compiles correctly."""
    try:
        with open(filepath) as f:
            source = f.read()
        compile(source, filepath, 'exec')
        return True
    except SyntaxError as e:
        print(f"  SYNTAX ERROR in {filepath}: {e}")
        return False


def main():
    dry_run = '--run' not in sys.argv
    revert = '--revert' in sys.argv
    
    if revert:
        print("Reverting all target files to HEAD...")
        files_str = ' '.join(TARGET_FILES)
        os.system(f'cd {os.path.dirname(os.path.dirname(__file__))} && git checkout -- {files_str}')
        print("Reverted.")
        return
    
    print(f"Print-to-Logger Migration v2 ({'DRY RUN' if dry_run else 'LIVE'})")
    print("=" * 60)
    
    total_replacements = 0
    migrated_count = 0
    
    for fp in TARGET_FILES:
        full_path = fp if os.path.exists(fp) else os.path.join(os.path.dirname(os.path.dirname(__file__)), fp)
        count = migrate_file(full_path, dry_run=dry_run)
        if count > 0:
            migrated_count += 1
            total_replacements += count
    
    print(f"\nSummary: {migrated_count} files affected, {total_replacements} total replacements")
    
    if not dry_run:
        # Verify syntax
        print("\nVerifying syntax...")
        errors = 0
        for fp in TARGET_FILES:
            full_path = fp if os.path.exists(fp) else os.path.join(os.path.dirname(os.path.dirname(__file__)), fp)
            if not verify_syntax(full_path):
                errors += 1
        if errors == 0:
            print(f"All {len(TARGET_FILES)} files pass syntax check")
        else:
            print(f"{errors} files have syntax errors")


if __name__ == '__main__':
    main()
