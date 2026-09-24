"""The VENDORED-PATCHes in randomly/randomly.py do not change Rm's results.

    pixi run python tests/vendor_patch.py

Fits upstream Rm verbatim (randomly/ at c423387, from git) and the patched copy
on the same synthetic counts and seed, then compares every fitted attribute.
Everything select_genes reads must match bit for bit; the cleaned X is
reassociated (Vs (Vs^T X)), so it only has to match to rounding.
"""
import importlib, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERBATIM = "c423387"  # "vendor: randomly at e8730f8, verbatim"


def load_verbatim():
    tmp = Path(tempfile.mkdtemp()) / "randomly_verbatim"
    tmp.mkdir()
    for f in ("__init__.py", "randomly.py", "visualization.py", "clustering.py", "palettes.py"):
        src = subprocess.run(["git", "-C", ROOT, "show", f"{VERBATIM}:randomly/{f}"],
                             check=True, capture_output=True, text=True).stdout
        # the plotting-only imports this env does not ship (the other two patches)
        src = (src.replace("from MulticoreTSNE import MulticoreTSNE", "MulticoreTSNE = None")
                  .replace("import seaborn as sns", "sns = None"))
        (tmp / f).write_text(src)
    sys.path.insert(0, str(tmp.parent))
    return importlib.import_module("randomly_verbatim").Rm


def fit(Rm, counts):
    np.random.seed(0)  # upstream shuffles with np.random.permutation
    rm = Rm()
    rm.preprocess(counts)
    rm.fit()
    return rm, rm.select_genes(fdr=0.001)


def main():
    sys.path.insert(0, str(ROOT))
    from randomly import Rm as Patched
    Verbatim = load_verbatim()

    rng = np.random.default_rng(1)
    n, g, k = 600, 1500, 5  # cells, genes, planted programmes
    rate = np.exp(rng.normal(0, 1, g)) * (1 + 3 * (rng.random((n, k)) @ (rng.random((k, g)) < 0.05)))
    counts = pd.DataFrame(rng.poisson(rate), index=[f"c{i}" for i in range(n)],
                          columns=[f"g{j}" for j in range(g)])

    a, ga = fit(Verbatim, counts)
    b, gb = fit(Patched, counts)
    assert ga == gb, "selected genes differ"
    for attr in ("L", "Lr", "V", "Vr", "L_mp", "lambda_c", "_s", "_sa", "_snl", "_snr", "Ls", "Vs"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    assert a.n_components == b.n_components
    assert a.components_genes.keys() == b.components_genes.keys()
    for j in a.components_genes:
        assert np.array_equal(a.components_genes[j], b.components_genes[j]), f"components_genes[{j}]"
    assert np.allclose(a.X, b.X, rtol=1e-10, atol=1e-10), "cleaned X"
    print(f"OK: {len(ga)} genes, {a.n_components} components, "
          f"max |dX| = {np.abs(a.X - b.X).max():.2e}")


if __name__ == "__main__":
    main()
