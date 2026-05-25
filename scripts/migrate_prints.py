"""Migrate print() → logger.info/error/warning in specific files.
Usage: uv run python scripts/migrate_prints.py <file1.py> [file2.py ...]
Dry run: uv run python scripts/migrate_prints.py --dry-run <file1.py> ...

Converts:
  print(f"Processing {n} signals...") → logger.info("Processing %s signals...", n)
  print("[ERROR] Something broke")   → logger.error("Something broke")
  print("[WARN] Something odd")      → logger.warning("Something odd")
  print("[INFO] Something happened") → logger.info("Something happened")
"""
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def needs_logger(filepath: Path) -> bool:
    content = filepath.read_text(encoding='utf-8')
    # Check if logging is imported and logger is defined
    has_logging_import = 'import logging' in content
    has_logger_def = 'getLogger' in content
    return has_logging_import or has_logger_def

def ensure_logger(filepath: Path) -> bool:
    """Add logger = logging.getLogger(__name__) if missing but logging is imported."""
    content = filepath.read_text(encoding='utf-8')
    if 'import logging' in content and 'getLogger' not in content:
        # Find where to insert logger definition (after last import)
        lines = content.split('\n')
        insert_after = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                insert_after = i
        for i in range(insert_after + 1, min(insert_after + 5, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith(('import ', 'from ')):
                insert_after = i
            elif stripped == '' or stripped.startswith('#'):
                continue
            else:
                break
        lines.insert(insert_after + 1, '')
        lines.insert(insert_after + 2, 'logger = logging.getLogger(__name__)')
        filepath.write_text('\n'.join(lines), encoding='utf-8')
        return True
    return False

def convert_print(line: str) -> str | None:
    """Convert a print() line to logger call. Returns None if no conversion needed."""
    stripped = line.lstrip()
    indent = line[:len(line) - len(stripped)]
    
    # Match: print("some text")  or  print(f"some {var} text")
    m = re.match(r'print\((.+)\)', stripped)
    if not m:
        return None
    
    args = m.group(1)
    
    # Determine log level from prefixes
    level = 'info'
    if '[ERROR]' in args or '[ERR]' in args:
        level = 'error'
        args = args.replace('[ERROR] ', '').replace('[ERR] ', '')
    elif '[WARN]' in args or '[WARNING]' in args:
        level = 'warning'
        args = args.replace('[WARN] ', '').replace('[WARNING] ', '')
    elif '[INFO]' in args or '[INF]' in args:
        level = 'info'
        args = args.replace('[INFO] ', '').replace('[INF] ', '')
    elif '[DEBUG]' in args:
        level = 'debug'
        args = args.replace('[DEBUG] ', '')
    
    # Handle f-strings: convert {var} → %s and add variables as args
    f_string = args.startswith('f"') or args.startswith("f'")
    
    if f_string:
        # Find all {var} substitutions
        fmt_string = args[2:]  # remove f" or f'
        quote_char = fmt_string[0]  # " or '
        
        # Extract the content between the quotes
        # Handle nested braces
        content = ''
        i = 0
        depth = 0
        start_char = None
        while i < len(fmt_string):
            ch = fmt_string[i]
            if i == 0:
                start_char = ch
                i += 1
                continue
            if ch == '\\' and i + 1 < len(fmt_string):
                content += ch + fmt_string[i + 1]
                i += 2
                continue
            if ch == start_char and depth == 0:
                break
            if ch == '{':
                depth += 1
                if depth == 1:
                    content += '%s'
                else:
                    content += ch
            elif ch == '}':
                depth -= 1
                if depth < 0:
                    content += ch
                    depth = 0
                elif depth > 0:
                    content += ch
            else:
                if depth > 0 and depth == 1:
                    pass  # part of the variable expression
                elif depth == 0:
                    content += ch
                else:
                    content += ch
            i += 1
        
        # Extract variable names from f-string
        # For simple cases like {name}, {n}, {symbol}
        vars_list = re.findall(r'\{([^}]+)\}', fmt_string)
        # Clean up variable expressions to just the first identifier
        clean_vars = []
        for v in vars_list:
            v = v.strip()
            # Handle format specifiers: {var:6.1%} → var
            v = v.split(':')[0] if ':' in v else v
            # Handle method calls: {symbol:6} → symbol
            v = v.split(' ')[0] if ' ' in v else v
            clean_vars.append(v)
        
        if clean_vars:
            var_str = ', '.join(clean_vars)
            return f'{indent}logger.{level}("{content}", {var_str})'
        else:
            return f'{indent}logger.{level}("{content}")'
    else:
        # Non-f-string: just wrap the string content
        # Strip the outer quotes to get the message
        inner = args.strip()
        if inner.startswith(('"', "'")) and (inner.endswith('"') or inner.endswith("'")):
            quote = inner[0]
            # Find matching end quote
            end_idx = inner.rfind(quote)
            if end_idx > 0:
                inner_content = inner[1:end_idx]
                rest = inner[end_idx+1:].strip()
                if rest.startswith(','):
                    # print("msg", var) - this format needs different handling
                    var_part = rest[1:].strip()
                    return f'{indent}logger.{level}("{inner_content} %s", {var_part})'
                return f'{indent}logger.{level}("{inner_content}")'
        
        # Fallback: wrap as-is
        return f'{indent}logger.{level}({args})'


def migrate_file(filepath: Path, dry_run: bool = False) -> int:
    """Migrate print() → logger calls in a file. Returns number of conversions."""
    original = filepath.read_text(encoding='utf-8')
    lines = original.split('\n')
    converted = 0
    
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('print(') and not stripped.startswith('# print('):
            # Check it's not inside a comment or string
            result = convert_print(line)
            if result:
                new_lines.append(result)
                converted += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    new_content = '\n'.join(new_lines)
    
    if converted > 0 and not dry_run:
        filepath.write_text(new_content, encoding='utf-8')
        print(f"  {filepath.name}: {converted} conversions")
    elif converted > 0 and dry_run:
        print(f"  {filepath.name}: {converted} would be converted (dry run)")
    else:
        print(f"  {filepath.name}: no conversions needed")
    
    return converted


def main():
    dry_run = '--dry-run' in sys.argv
    
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not files:
        print("Usage: uv run python scripts/migrate_prints.py [--dry-run] <file1.py> [file2.py ...]")
        # Default: scan for files needing migration
        print("\nScanning for files needing migration...")
        results = []
        for root, dirs, files in os.walk(str(PROJECT_ROOT / 'src')):
            for f in files:
                if f.endswith('.py'):
                    p = Path(root) / f
                    c = p.read_text(encoding='utf-8')
                    prints = c.count('print(')
                    # Only count actual print( calls, not in comments
                    loggers = c.count('logger.')
                    if prints > 5 and loggers == 0:
                        results.append((prints, p))
        results.sort(key=lambda x: -x[0])
        for prints, p in results:
            print(f"  {p.relative_to(PROJECT_ROOT)}: {prints} prints, no logger usage")
        return
    
    total = 0
    for f in files:
        fp = Path(f) if Path(f).is_absolute() else PROJECT_ROOT / f
        if not fp.exists():
            print(f"WARNING: {fp} not found")
            continue
        
        if not needs_logger(fp):
            print(f"SKIP: {fp.name} has no logging import — adding...")
            if not dry_run:
                ensure_logger(fp)
        
        n = migrate_file(fp, dry_run)
        total += n
    
    print(f"\nTotal: {total} conversions" if not dry_run else f"\nTotal: {total} would be converted")


if __name__ == '__main__':
    main()
