# omni-randomly

FEAT stage arm: Random Matrix Theory gene selection with
[RabadanLab/randomly](https://github.com/RabadanLab/randomly) (Aparicio et al.,
Patterns 2020).

Fits on raw counts (`rawdata_h5ad` `layers["counts"]`, subset to the cells and
genes of `normalized_h5`), selects genes at `--fdr`, writes `normalized_h5`
restricted to them. The number of genes is decided by the method.

`randomly/` is upstream `e8730f8` vendored, patched so the MulticoreTSNE and
seaborn imports are optional (see `randomly/UPSTREAM`).

The shared args (`--output_dir`, `--name` and the FEAT stage I/O) come from the
plan's JSON schemas shipped in `src/common/`; `pixi run -e dev sync` refreshes them.

```bash
pixi run check
pixi run python feat-select.py --output_dir out --name <dataset> \
  --rawdata_h5ad <d>.h5ad --normalized_h5 <d>_normalized.h5 \
  --filtered_cellids <d>_cellids.txt.gz --filtered_featureids <d>_featureids.txt.gz \
  --properties_info <d>_properties.yaml --fdr 0.001
```

`--backend cupy` runs the eigh on the GPU (`pixi run -e gpu ...`, linux-64,
float64). It reproduces the CPU gene set exactly; at 5k cells it is not faster.
`randomly_fast.py` (`RmFast`) also replaces upstream's per-column shuffle loop;
`pixi run -e gpu agreement` checks it against upstream `Rm` and CPU vs GPU.

Cost is O(n_cells²) memory and O(n_cells³) time; CPU runs above 40000 cells
are refused (`MAX_CELLS`). Measured fit on tenx-0020k subsamples (~17k genes):

| cells | fit time | peak RSS |
|---|---|---|
| 5 000 | 27 s | 4.0 GB |
| 10 000 | 2.8 min | 9.0 GB |
| 15 000 | 8.1 min | 15.4 GB |
| 40 000 (projected) | ~2.5 h | ~60 GB |
