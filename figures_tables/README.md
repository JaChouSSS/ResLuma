# ResLuma figure and table reproduction package

This standalone ZIP contains the reproduction scripts, table data, figure data/image tiles, original numerical sources, and generated figure outputs for the ResLuma manuscript. It intentionally contains **no LaTeX manuscript chapters, `main.tex`, or author version of `main.pdf`**.

- Figure 1 and Figures 3-19: Python scripts in `scripts/` and outputs in `generated_figures/`.
- Figure 2: the author's latest manually updated framework PDF at `figures_original_pdf/Figure_02.pdf`. No Visio editable source or Python reconstruction script is supplied for this manually prepared figure.
- Tables 1-9: source CSV files and Table 6 LaTeX template in `tables/`. In case of display/format discrepancies, the **LaTeX manuscript tables are authoritative**.
- `inputs/tiles/` consists of image tiles extracted from earlier figures; regenerating these composite figures does not rerun the underlying trained LLIE methods.

Requirements and commands (run in the extracted top-level directory):

```bash
pip install -r requirements.txt
python run_all.py
python render_tables_markdown.py
```

The 4K/8K inference measurements are original experimental data; regenerating plots does not rerun these GPU benchmarks. The independent curve benchmark in `benchmarks/` is not claimed to reproduce the original CPU measurements exactly.

The accompanying paper source and compiled PDF are distributed in the **separate** `ResLuma_LaTeX_With_Authors.zip`.
