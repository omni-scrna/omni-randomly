"""RmFast agrees with upstream Rm, and its cupy backend with its numpy one.

    pixi run -e gpu python tests/agreement.py                          # synthetic smoke
    pixi run -e gpu python tests/agreement.py NORMALIZED.h5            # real data

The synthetic data is a smoke test only: its planted programmes leave no genes
near the FDR threshold, so it cannot see float32 drift. Real data can
(tenx-0005k: float32 moved 5 of 3805 genes), so run the second form before
changing RmFast.

1. RmFast(numpy) vs upstream Rm: data eigenvalues identical (same scipy eigh,
   no randomness); gene set within upstream's own seed-to-seed spread, since
   the shuffle stream differs by construction.
2. RmFast(cupy, float64) vs RmFast(numpy), same shuffle: identical gene set.
3. RmFast(cupy, float32): same n_components, gene-set Jaccard >= 0.99.
GPU checks are skipped without a CUDA device.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from randomly import Rm
from randomly_fast import RmFast, has_gpu

FDR = 0.001


def make_counts(n_cells=3000, n_genes=1000, seed=0):
    """Poisson noise plus 6 planted cell-group x gene-block programmes."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float64)
    for _ in range(6):
        c = rng.choice(n_cells, size=n_cells // 8, replace=False)
        g = rng.choice(n_genes, size=n_genes // 10, replace=False)
        X[np.ix_(c, g)] += rng.poisson(15.0, size=(len(c), len(g)))
    return pd.DataFrame(X, index=[f"c{i}" for i in range(n_cells)],
                        columns=[f"g{j}" for j in range(n_genes)])


def real_normalized(normalized_h5):
    spec = importlib.util.spec_from_file_location("ndimr", ROOT / "ndimr.py")
    nd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nd)
    return nd.read_tenx(normalized_h5)


def run(rm, df, np_seed=None, counts=True):
    if np_seed is not None:
        np.random.seed(np_seed)  # upstream's shuffle draws from the global RNG
    if counts:
        rm.preprocess(df.copy())
        rm.fit()
    else:  # already normalized: upstream's non-preprocessed branch
        rm.fit(df.copy())
    return rm, set(rm.select_genes(fdr=FDR))


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    counts = len(sys.argv) != 2
    df = make_counts() if counts else real_normalized(sys.argv[1])
    print(f"data: {df.shape[0]} cells x {df.shape[1]} genes")

    ref0, g_ref0 = run(Rm(), df, np_seed=0, counts=counts)
    ref1, g_ref1 = run(Rm(), df, np_seed=1, counts=counts)
    cpu, g_cpu = run(RmFast(backend="numpy", shuffle_seed=0), df, counts=counts)
    spread = jaccard(g_ref0, g_ref1)
    print(f"upstream seed 0 vs 1: n_comp {ref0.n_components}/{ref1.n_components}, "
          f"genes {len(g_ref0)}/{len(g_ref1)}, jaccard {spread:.4f}")
    print(f"RmFast-numpy vs upstream: n_comp {cpu.n_components}, genes {len(g_cpu)}, "
          f"jaccard {jaccard(g_cpu, g_ref0):.4f}")
    assert np.array_equal(cpu.L, ref0.L), "data eigenvalues must not depend on the shuffle"
    assert len(g_ref0) > 0, "no signal genes selected"
    assert jaccard(g_cpu, g_ref0) >= min(spread, 0.95), "RmFast outside upstream's seed spread"

    if not has_gpu():
        print("no CUDA device: GPU agreement skipped")
        return
    for dtype in ("float64", "float32"):
        gpu, g_gpu = run(RmFast(backend="cupy", dtype=dtype, shuffle_seed=0), df, counts=counts)
        rel = np.max(np.abs(gpu.L - cpu.L)) / np.max(np.abs(cpu.L))
        print(f"RmFast-cupy {dtype} vs numpy: max rel eig err {rel:.2e}, n_comp "
              f"{gpu.n_components}/{cpu.n_components}, genes {len(g_gpu)}/{len(g_cpu)}, "
              f"jaccard {jaccard(g_gpu, g_cpu):.4f}")
        assert gpu.n_components == cpu.n_components
        if dtype == "float64":
            assert g_gpu == g_cpu, f"float64 gene sets differ in {len(g_gpu ^ g_cpu)} genes"
        else:
            assert jaccard(g_gpu, g_cpu) >= 0.99, "float32 drift beyond 1% of the gene set"
    print("OK")


if __name__ == "__main__":
    main()
