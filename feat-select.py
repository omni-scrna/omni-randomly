#!/usr/bin/env python3
"""FEAT module: Random Matrix Theory gene selection (RabadanLab/randomly).

Randomly log2(1+TPM)-transforms raw counts, eigendecomposes the cells x cells
Wishart matrix, fits Marchenko-Pastur to a column-shuffled copy and keeps the
genes whose projection onto the above-Tracy-Widom components beats the noise
at the requested FDR. The gene count is the method's output, not an input, so
there is no --number_selected.

Counts come from rawdata_h5ad layers["counts"], subset to the cells and genes
of --normalized_h5 (the fe-scanpy pearson_residuals convention), so FILT's
filter applies and the NORM arm does not change which genes are chosen. The
written values are normalized_h5 restricted to those genes.

Cost is O(n_cells^2) memory and O(n_cells^3) time, twice (data + shuffle).
"""

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from randomly import Rm

# ponytail: CPU ceiling, two dense n^2 float64 Wisharts + full eigh. Lift when
# the RmFast (cupy) GPU arm lands.
MAX_CELLS = 20000


def parse_args():
    p = argparse.ArgumentParser(description="FEAT module: randomly")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rawdata_h5ad", nargs="+", required=True)
    p.add_argument("--normalized_h5", nargs="+", required=True)
    p.add_argument("--filtered_cellids", nargs="+", required=True)
    p.add_argument("--filtered_featureids", nargs="+", required=True)
    p.add_argument("--properties_info", nargs="+", required=True)
    p.add_argument("--fdr", type=float, required=True,
                   help="false discovery rate for signal genes, in (0, 1)")
    return p.parse_args()


def read_tenx(path):
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][:])
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        genes = g["genes"][:].astype(str)
        cells = g["barcodes"][:].astype(str)
    return m, genes, cells  # genes x cells


def write_tenx(path, m, genes, cells):
    m = sparse.csc_matrix(m)
    m.sort_indices()
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"] = m.data
        g["indices"] = m.indices.astype(np.uint32)
        g["indptr"] = m.indptr.astype(np.int64)
        g["shape"] = np.array(m.shape, dtype=np.uint32)
        g["genes"] = np.array(genes, dtype="S")
        g["barcodes"] = np.array(cells, dtype="S")


def select_genes(counts, fdr):
    """counts: cells x genes DataFrame of raw counts -> list of selected genes."""
    # ponytail: fixed seed. Rm ignores its random_state; the shuffle behind the
    # MP fit draws from numpy's global RNG and moves lambda_c, hence the genes.
    # TODO: expose as a random_seed plan parameter for the seed factorial.
    np.random.seed(0)
    rm = Rm()
    rm.preprocess(counts)
    rm.fit()
    print(f"signal components above Tracy-Widom: {rm.n_components}")
    return rm.select_genes(fdr=fdr)


def main():
    args = parse_args()
    if not 0 < args.fdr < 1:
        raise ValueError(f"--fdr must be in (0, 1), got {args.fdr}")

    norm, genes, cells = read_tenx(args.normalized_h5[0])
    print(f"normalized_h5 (genes x cells): {norm.shape}")
    if len(cells) > MAX_CELLS:
        raise ValueError(f"{len(cells)} cells > MAX_CELLS={MAX_CELLS}: the n_cells^2 "
                         "Wishart does not fit on CPU; use the GPU arm")

    raw = ad.read_h5ad(args.rawdata_h5ad[0])
    if "counts" not in raw.layers:
        raise ValueError("randomly needs raw counts in rawdata_h5ad layers['counts']")
    x = raw[cells, genes].layers["counts"]
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    counts = pd.DataFrame(x, index=cells, columns=genes)

    selected = set(select_genes(counts, args.fdr))
    keep = np.array([g in selected for g in genes])
    print(f"selected {keep.sum()} / {len(genes)} genes at fdr={args.fdr}")
    if not keep.any():
        raise ValueError("no genes selected: no signal above the Marchenko-Pastur edge at this fdr")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tenx(out / f"{args.name}_normalized_selected.h5", norm[keep, :], genes[keep], cells)


if __name__ == "__main__":
    main()
