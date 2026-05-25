#!/usr/bin/env python3
"""
Print-to-Logger Migration — Phase 2, v3
Uses Python's AST to safely find module-level import locations.
"""
import ast
import os
import sys

TARGET_FILES = [
    # Phase 4: Top non-main() print offenders
    'src/monitor/unified_dashboard.py',
    'src/backtest/alternative_data_backtest.py',
    'src/backtest/cross_asset_regime_arb_backtest.py',
    'src/backtest/duration_yield_backtest.py',
    'src/backtest/multi_speed_momentum_backtest.py',
    'src/strategy/adaptive_sizing.py',
    'src/backtest/vixy_hedge_backtest.py',
    'src/backtest/unified_overlay_backtest.py',
    'src/backtest/bond_duration_backtest.py',
    'src/signals/vpin_bvc.py',
    'src/monitor/performance_attribution.py',
    'src/strategy/graduation_checklist.py',
    'src/signals/cross_asset_relative_value.py',
    'src/signals/cross_asset_regime_arb.py',
    'src/backtest/ensemble_backtest.py',
]

# Files where print() calls in main() are legit CLI output — skip those lines
# We still migrate non-main() prints if any exist
CLI_FILES = {
    'src/strategy/vol_parity_allocator.py',
    'src/backtest/dbc_weight_sweep.py',
    'src/backtest/car25.py',
    'src/backtest/combined_strategy.py',
    'src/backtest/run_actual_ubt_validation.py',
}


def get_last_top_level_import_line(source: str) -> int:
    """Find the last line number of a top-level import using AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1
    
    last_import_line = -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only top-level imports (parent is the module itself)
            if hasattr(node, 'col_offset') and node.col_offset == 0:
                # Get the end line
                end_line = getattr(node, 'end_lineno', node.lineno)
                if end_line > last_import_line:
                    last_import_line = end_line
    return last_import_line


def has_logger(content: str) -> bool:
    """Check if module already has logging setup."""
    return 'logger = logging.getLogger' in content


def has_logging_import(content: str) -> bool:
    """Check if module already imports logging."""
    return 'import logging' in content


def add_logger_boilerplate(content: str) -> str:
    """Add import logging and logger = getLogger at module level after last import."""
    if has_logger(content):
        return content
    
    lines = content.split('\n')
    
    # Find the last top-level import via AST
    insert_line = get_last_top_level_import_line(content)
    
    if insert_line < 0:
        # Fallback: insert after module docstring
        insert_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('"""', "'''")):
                # Skip until docstring closes
                if stripped.endswith(('"""', "'''")) and len(stripped) > 3:
                    insert_line = i + 1
                    break
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().endswith(('"""', "'''")):
                        insert_line = j + 1
                        break
                break
            elif stripped and not stripped.startswith('#'):
                insert_line = i
                break
    
    # Skip blank lines after insertion point
    while insert_line < len(lines) and lines[insert_line].strip() == '':
        insert_line += 1
    
    # Insert logger block
    logger_block = []
    if not has_logging_import(content):
        logger_block.append('import logging')
        logger_block.append('')
    logger_block.append('logger = logging.getLogger(__name__)')
    logger_block.append('')
    
    new_lines = lines[:insert_line] + logger_block + lines[insert_line:]
    return '\n'.join(new_lines)


def is_in_main_block(lines: list, line_idx: int) -> bool:
    """Check if given line is inside a main() function or __main__ guard."""
    depth = 0
    in_main = False
    for i in range(line_idx + 1):
        stripped = lines[i].strip()
        if stripped.startswith('def main('):
            in_main = True
            depth = 0
        elif in_main and stripped.startswith('def ') and not stripped.startswith('def main('):
            # A new function definition — only reset if at same indent level as main
            indent = len(lines[i]) - len(lines[i].lstrip())
            if indent <= 4:  # main is usually at indent 0, body at indent 4
                in_main = False
        elif stripped == "if __name__ == '__main__':" or stripped == 'if __name__ == "__main__":':
            in_main = True
    
    if not in_main:
        return False
    
    # Check if this particular line is inside main (not function definitions inside main)
    stripped = lines[line_idx].strip()
    return 'print(' in stripped


def migrate_file(filepath: str, dry_run: bool = True) -> int:
    """Migrate print() to logger.info() in a file."""
    if not os.path.exists(filepath):
        # Try relative to project root
        cwd = os.getcwd()
        filepath = os.path.join(cwd, filepath)
        if not os.path.exists(filepath):
            print(f"  MISSING: {filepath}")
            return 0
    
    with open(filepath) as f:
        content = f.read()
    
    original = content
    total_prints = content.count('print(')
    if total_prints == 0:
        print(f"  ✓ NO PRINTS: {filepath}")
        return 0
    
    # Step 1: Add logger boilerplate
    content = add_logger_boilerplate(content)
    
    # Step 2: Replace print( with logger.info(
    # Preserve print() calls inside main() for CLI tools
    lines = content.split('\n')
    new_lines = []
    replacements = 0
    
    in_main_func = False
    main_indent = -1
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        orig_line = line
        
        # Track main() function scope via indentation
        if stripped.startswith('def main('):
            in_main_func = True
            main_indent = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue
        
        if in_main_func:
            indent = len(line) - len(line.lstrip())
            # Check if we've exited main (next function at same level as main, or end of file)
            if indent <= main_indent and stripped.startswith(('def ', 'class ', '@')):
                in_main_func = False
            elif stripped == '' and i < len(lines) - 1:
                pass  # blank line, could still be in main
            elif indent <= main_indent and stripped and not stripped.startswith('#'):
                in_main_func = False
        
        # Check if we're in a __main__ guard
        if stripped == "if __name__ == '__main__':" or stripped == 'if __name__ == "__main__":':
            in_main_func = True
            main_indent = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue
        
        # Skip print() in main() — those are CLI outputs
        if in_main_func and 'print(' in stripped and not stripped.startswith('#'):
            new_lines.append(line)
            continue
        
        # Replace print( with logger.info( in non-CLI code
        if 'print(' in stripped and not stripped.startswith('#'):
            # Skip __repr__/__str__ methods
            if '__repr__' in stripped or '__str__' in stripped:
                new_lines.append(line)
                continue
            
            # Check not inside a string
            new_line = line.replace('print(', 'logger.info(')
            count = line.count('print(')
            replacements += count
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content == original:
        print(f"  - NO CHANGE: {filepath} (all {total_prints} prints in main() or CLI)")
        return 0
    
    # Verify syntax BEFORE writing
    try:
        compile(content, filepath, 'exec')
    except SyntaxError as e:
        print(f"  ✗ SYNTAX ERROR would result: {filepath}: {e}")
        print(f"    Reverting...")
        return -1
    
    if dry_run:
        print(f"  → WOULD MIGRATE: {filepath} ({total_prints} prints -> logger, {replacements} replacements)")
        # Show a few key diffs
        orig_lines = original.split('\n')
        new_lines = content.split('\n')
        shown = 0
        for idx, (o, n) in enumerate(zip(orig_lines, new_lines)):
            if o != n and shown < 10:
                if 'logger.info' in n and 'print(' in o:
                    print(f"    L{idx+1}: print( -> logger.info(")
                    shown += 1
        return 0
    else:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ MIGRATED: {filepath} ({total_prints} prints -> logger, {replacements} replacements)")
        return replacements


def main():
    dry_run = '--run' not in sys.argv
    
    print(f"Print-to-Logger Migration v3 ({'DRY RUN' if dry_run else 'LIVE'})")
    print("=" * 60)
    
    cwd = os.getcwd()
    results = []
    
    for fp in TARGET_FILES:
        full_path = fp if os.path.exists(fp) else os.path.join(cwd, fp)
        result = migrate_file(full_path, dry_run=dry_run)
        results.append((fp, result))
    
    migrated = sum(1 for _, r in results if r > 0)
    total_repl = sum(max(0, r) for _, r in results)
    errors = sum(1 for _, r in results if r < 0)
    
    print(f"\nSummary: {migrated} files to migrate, {total_repl} total replacements, {errors} errors")
    
    if errors > 0:
        print("\nFiles with errors:")
        for fp, r in results:
            if r < 0:
                print(f"  - {fp}")


if __name__ == '__main__':
    main()
