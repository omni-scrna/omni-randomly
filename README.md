# omni-randomly

NDIMR stage arm (normalized dimensionality reduction): Random Matrix Theory
denoising with [RabadanLab/randomly](https://github.com/RabadanLab/randomly)
(Aparicio et al., Patterns 2020).

Fits on the full normalized matrix (`normalized_h5`, all genes, no feature
selection) and keeps the components above the Tracy-Widom edge of a
Marchenko-Pastur fit to a column-shuffled copy. The number of components is
decided by the method. That is PCA on the gene correlation matrix, cut where the
noise ends, so it writes what the PCA stage writes: `{name}_embedding.tsv`
(`cell_id`, `PC1..PCk`) and `{name}_loadings.tsv` (`gene`, `PC1..PCk`);
`tests/pca_equivalence.py` checks both against sklearn's PCA.

`randomly/` is upstream `e8730f8` vendored and patched (see `randomly/UPSTREAM`).

The shared args (`--output_dir`, `--name` and the NDIMR stage I/O) come from the
plan's JSON schemas shipped in `src/common/`; `pixi run -e dev sync` refreshes them.

```bash
pixi run check
pixi run python ndimr.py --output_dir out --name <dataset> \
  --normalized_h5 <d>_normalized.h5 --random_seed 42
```

`--random_seed` seeds the column shuffle behind the Marchenko-Pastur fit.
`--backend cupy` runs the eigh on the GPU (`pixi run -e gpu ...`, linux-64,
float64) and reproduces the CPU result exactly. `randomly_fast.py` (`RmFast`)
also replaces upstream's per-column shuffle loop; `pixi run -e gpu agreement`
checks it against upstream `Rm` and CPU vs GPU.

tenx-0005k (4925 cells, 27998 genes, scanpy-normalized): 276 components, 43 s,
4.9 GB peak RSS.

Cost is O(n_cells²) memory and O(n_cells³) time; CPU runs above 40000 cells
are refused (`MAX_CELLS`). Measured fit on tenx-0020k subsamples (~17k genes):

| cells | fit time | peak RSS |
|---|---|---|
| 5 000 | 27 s | 4.0 GB |
| 10 000 | 2.8 min | 9.0 GB |
| 15 000 | 8.1 min | 15.4 GB |
| 40 000 (projected) | ~2.5 h | ~60 GB |
