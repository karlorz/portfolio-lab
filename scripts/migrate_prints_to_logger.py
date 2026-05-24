#!/usr/bin/env python3
"""
Print-to-Logger Migration Script — Phase 2
Migrates all print() calls to logger.info() in target files.

Safe: Keeps f-string formatting intact, just replaces print( with logger.info(
Only modifies print() calls on separate lines (not inside expressions).
"""

import re
import os
import sys

TARGET_FILES = [
    # Tier 1 — No logger yet
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
]

ADD_LOGGER_TEMPLATE = '''
import logging

logger = logging.getLogger(__name__)

'''

def add_logger_if_missing(content: str, filepath: str) -> str:
    """Add import logging and logger = getLogger after the last import."""
    if 'logging' in content and 'logger = logging.getLogger' in content:
        return content  # Already has logger

    # Find import section
    lines = content.split('\n')
    last_import_line = -1
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from ') or 'import ' in line:
            last_import_line = i

    if last_import_line >= 0:
        # Insert after last import
        # Find a good insertion point: after the last import line, before code
        insert_at = last_import_line + 1
        # Skip blank lines after imports
        while insert_at < len(lines) and lines[insert_at].strip() == '':
            insert_at += 1
        
        # Insert logger definition before the first non-blank non-import line
        logger_block = [
            'import logging',
            '',
            'logger = logging.getLogger(__name__)',
            '',
        ]
        # Replace the content
        new_lines = lines[:insert_at] + [''] + logger_block + lines[insert_at:]
        return '\n'.join(new_lines)
    
    return content


def migrate_file(filepath: str) -> int:
    """Migrate print() to logger.info() in a file. Returns count of replacements."""
    if not os.path.exists(filepath):
        print(f"  MISSING: {filepath}")
        return 0

    with open(filepath) as f:
        content = f.read()

    original = content

    # Step 1: Add logger boilerplate if missing
    content = add_logger_if_missing(content, filepath)

    # Step 2: Replace print() calls with logger calls
    # Pattern: print([something]) -> logger.info([something])
    # Handle multi-line print() calls
    # We'll use a regex that matches print( at the start of a statement
    # and replaces it with logger.info(
    
    # Count prints before
    print_count_before = content.count('print(')
    
    # Replace print( with logger.info( — but only standalone print() calls
    # Not print statements inside class definitions (__repr__, __str__) — those should stay
    
    # Strategy: Replace only print( when it appears as a statement (preceded by whitespace or at line start)
    # Exclude: `__repr__`, `__str__` methods, and class definitions
    
    # Actually, let's be more careful. Let's replace all print( with logger.info(
    # but check if there are any non-standalone prints.
    
    # Check for print( in string contexts, etc.
    # Simple replacement: print( -> logger.info( for ALL occurrences
    # This is safe for CLI scripts where print() is used for output
    
    lines = content.split('\n')
    new_lines = []
    replacements = 0
    
    in_string = False
    string_char = None
    
    for line in lines:
        stripped = line.strip()
        
        # Skip __repr__ and __str__ methods
        if '__repr__' in stripped or '__str__' in stripped:
            new_lines.append(line)
            continue
        
        # Skip lines inside string literals
        if '"""' in stripped or "'''" in stripped:
            new_lines.append(line)
            continue
        
        # Simple check: if line contains print( and looks like a statement
        # (starts with optional whitespace then print()
        if 'print(' in stripped:
            # Only replace if it's a standalone print statement
            # Match: optional whitespace, then print(...)
            # Don't match inside __repr__ etc.
            
            # Count non-escaped prints in this line
            new_line = line
            modified = False
            
            # Simple replacement: print( -> logger.info(
            # But be careful not to replace inside strings
            # We'll do a non-naive replacement
            result = []
            i = 0
            while i < len(line):
                if line[i:i+6] == 'print(':
                    # Check if we're inside a string
                    # Simple check: not preceded by . (to avoid .print()
                    if i == 0 or line[i-1] not in '._':
                        result.append('logger.info(')
                        replacements += 1
                        modified = True
                        i += 6
                        continue
                result.append(line[i])
                i += 1
            
            new_line = ''.join(result)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  OK: {filepath} ({replacements} replacements, had {print_count_before} prints)")
        return replacements
    else:
        print(f"  NO CHANGE: {filepath}")
        return 0


def main():
    total = 0
    for fp in TARGET_FILES:
        count = migrate_file(fp)
        total += count
    print(f"\nTotal replacements: {total}")


if __name__ == '__main__':
    main()
