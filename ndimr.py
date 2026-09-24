#!/usr/bin/env python3
"""NDIMR module: Random Matrix Theory dimensionality reduction (RabadanLab/randomly).

Fits randomly on the full normalized matrix (all genes, no feature selection):
z-scores each gene, eigendecomposes the cells x cells Wishart Y = X X^T / g, fits
Marchenko-Pastur to a column-shuffled copy and keeps the components above the
Tracy-Widom edge. The number of components is the method's output, not an input.

That is PCA on the gene correlation matrix, cut where the noise ends. With
X = U S W^T: the eigenvectors of Y are U and its eigenvalues S^2 / g, so

    embedding (cells x k) = U S       = V_s sqrt(g L_s)
    loadings  (genes x k) = W         = X^T V_s / sqrt(g L_s)

(tests/pca_equivalence.py checks both against sklearn's PCA).

Cost is O(n_cells^2) memory and O(n_cells^3) time, twice (data + shuffle).
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from randomly_fast import RmFast

sys.path.insert(0, str(Path(__file__).parent / "src"))  # vendored `common` (src/common)
from common import cli  # noqa: E402

# ponytail: CPU ceiling. Measured on tenx-0020k subsamples (5k/10k/15k cells,
# ~17k genes): peak RSS ~ 2.4 n^2 + 5.8 n*g float64 (Wisharts + eigh vs. dense
# counts copies), fit time ~ n^3 (15k: 8 min). 40k cells ~ 60 GB, ~2.5 h, under
# an 80 GB job with room for more genes. The cupy backend is bounded by VRAM
# instead and fails loudly on OOM.
MAX_CELLS = 40000


def parse_args():
    p = argparse.ArgumentParser(description="NDIMR module: randomly")
    cli.add_base_args(p)              # --output_dir, --name
    cli.add_stage_args(p, "NDIMR")    # --normalized_h5, from the plan's schema
    p.add_argument("--random_seed", type=int, default=0,
                   help="seed of the column shuffle behind the Marchenko-Pastur fit")
    p.add_argument("--backend", choices=["numpy", "cupy"], default="numpy",
                   help="eigh on CPU (scipy) or GPU (cupy, float64)")
    return p.parse_args()


def read_tenx(path):
    """TENx H5 (genes x cells) -> cells x genes float64 DataFrame."""
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][:])
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        genes = g["genes"][:].astype(str)
        cells = g["barcodes"][:].astype(str)
    return pd.DataFrame(m.T.toarray().astype(np.float64), index=cells, columns=genes)


def fit(X, backend="numpy", seed=0):
    """X: cells x genes normalized DataFrame -> fitted RmFast.

    Skips upstream's fit(df), which drops genes whose column sum is <= 0: fine for
    counts, wrong for centred values such as pearson residuals. Constant genes
    are dropped instead, since the z-score divides by their std.
    """
    X = X.loc[:, X.std(axis=0).to_numpy() > 0]
    rm = RmFast(backend=backend, shuffle_seed=seed)
    rm.X = X.to_numpy()
    rm.n_cells, rm.n_genes = X.shape
    rm.normal_cells, rm.normal_genes = X.index, X.columns
    rm._fit()
    return rm


def embed(rm):
    """Signal components, largest first -> (embedding, loadings) DataFrames."""
    L, V = rm.Ls[::-1], rm.Vs[:, ::-1]  # eigh returns ascending
    scale = np.sqrt(rm.n_genes * L)
    # _fit leaves rm.X = V_s V_s^T X; its projection on V_s equals X^T V_s
    loadings = rm.X.T @ V / scale
    pcs = [f"PC{i + 1}" for i in range(len(L))]
    return (pd.DataFrame(V * scale, index=rm.normal_cells, columns=pcs),
            pd.DataFrame(loadings, index=rm.normal_genes, columns=pcs))


def main():
    args = parse_args()
    X = read_tenx(args.normalized_h5)
    print(f"normalized_h5 (cells x genes): {X.shape}")
    if args.backend == "numpy" and X.shape[0] > MAX_CELLS:
        raise ValueError(f"{X.shape[0]} cells > MAX_CELLS={MAX_CELLS}: the n_cells^2 "
                         "Wishart does not fit on CPU; use the GPU arm")

    rm = fit(X, args.backend, args.random_seed)
    print(f"signal components above Tracy-Widom: {rm.n_components}")
    if rm.n_components == 0:
        raise ValueError("no components above the Tracy-Widom edge: no signal to embed")
    embedding, loadings = embed(rm)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    embedding.to_csv(out / f"{args.name}_embedding.tsv", sep="\t", index_label="cell_id")
    loadings.to_csv(out / f"{args.name}_loadings.tsv", sep="\t", index_label="gene")


if __name__ == "__main__":
    main()
