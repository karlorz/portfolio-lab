"""Scan for files with excessive print() calls."""
import os

results = []
for root, dirs, files in os.walk('src'):
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            with open(p, encoding='utf-8', errors='replace') as fh:
                c = fh.read()
            prints = c.count('print(')
            loggers = c.count('logger.')
            if prints > 10:
                results.append((prints, loggers, p))

results.sort(key=lambda x: -x[0])
print('Files with >10 print() calls:')
for prints, loggers, p in results:
    status = 'NO LOGGER' if loggers == 0 else f'{loggers} logger calls'
    print(f'  {p}: {prints} print(), {status}')
print(f'\nTotal: {len(results)} files')
