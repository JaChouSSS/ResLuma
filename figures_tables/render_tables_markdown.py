#!/usr/bin/env python3
"""Create Markdown previews of all 9 tables from the manuscript CSV data."""
from pathlib import Path
import csv, json
R=Path(__file__).resolve().parent
index=json.loads((R/'tables'/'TABLE_INDEX.json').read_text(encoding='utf-8'))['tables']
out=R/'table_previews'
out.mkdir(exist_ok=True)
for no,file in index.items():
    with (R/'tables'/file).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f)
        data=list(reader)
    def fmt(row):
        return '| '+' | '.join(str(s).replace('|',r'\|') for s in row)+' |'
    content=[f'# Table {no} — {file}','',fmt(data[0]),fmt(['---']*len(data[0]))]
    content.extend(fmt(row) for row in data[1:])
    (out/f'Table_{int(no):02d}.md').write_text('\n'.join(content)+'\n',encoding='utf-8')
    print(f'Table {no}: {len(data)-1} data rows')
