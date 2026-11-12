#!/usr/bin/env python3
"""Regenerate all 18 manuscript figures, excluding Figure 2 framework. """
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).parent
for n in range(1,20):
    if n==2:continue
    script=ROOT/'scripts'/f'figure_{n:02d}.py'
    print(f'[{n:02d}]',flush=True)
    subprocess.run([sys.executable,str(script)],cwd=ROOT,check=True)
print('All 18 figures finished in generated_figures/')
