# Paper assets

This directory contains the exact active figure files referenced by the final `main.tex` files in the two Overleaf archives uploaded for the papers.

The former `SSCP1.png` Springer/SSVM webpage screenshot was removed because it is not needed for code reproducibility.

## Contents

```text
paper_assets/
├── sure_ssvm2025/
│   ├── reconstructions.png
│   ├── coveragedenoisingwithoracle.png
│   ├── coveragedeblurringwithoracle.png
│   ├── histogramdenoising.png
│   ├── histogramdeblurring.png
│   └── ACTIVE_LATEX_FIGURES.txt
├── poisson_ssp2025/
│   ├── poisson_reconstruction_reviewed.png
│   ├── coveragepoissondenoisingwithoracle.png
│   ├── coveragepoissondeblurringwithoracle.png
│   └── ACTIVE_LATEX_FIGURES.txt
└── MANIFEST.json
```

The file `MANIFEST.json` records the source LaTeX path, the corresponding paper figure, file size, and SHA256 checksum. Numerical arrays and regenerated reference plots are kept separately under `results/reference/`.

Commented-out `\includegraphics` entries from the LaTeX sources are intentionally not included here.
