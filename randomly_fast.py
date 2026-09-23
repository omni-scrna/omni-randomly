# Copyright (c) 2026 ben carrillo. GPU port of randomly.Rm (RmFast).
# SPDX-License-Identifier: MIT
# Subclasses randomly.Rm, Copyright (c) 2018 Luis Aparicio, MIT (see randomly/LICENSE).
"""`RmFast`: `randomly.Rm` with a GPU eigh and a vectorised shuffle.

Overrides exactly two methods; everything else (MP fit, Tracy-Widom, gene
selection) is upstream's and runs on host arrays.

  * `_get_eigen(Y)`: `cupy.linalg.eigh` on the n_cells x n_cells Wishart
    (backend='cupy'), results copied back as float64. Same O(n^3) algorithm,
    faster hardware. backend='numpy' keeps the parent's scipy eigh.
  * `_random_matrix(X)`: an independent uniform permutation per column via
    argsort of uniform keys, instead of a Python-level loop. Distributionally
    identical to upstream, but drawn from its own seeded Generator, so a given
    seed gives the SAME shuffle on both backends and a DIFFERENT one from
    upstream's `np.random.permutation`.

dtype sets the GPU eigh precision. float64 (default) reproduces the numpy
backend's gene set exactly. float32 halves VRAM but is NOT exact: on
tenx-0005k it moved 5 of 3805 genes sitting on the FDR threshold (Jaccard
0.9987, same n_components). See tests/agreement.py.

Moved from kang2018_denoise/randomly_fast.py; the pip-CUDA dlopen workaround is
gone because conda-forge cupy ships cusolver/cublas in the env.
"""
import numpy as np

from randomly import Rm

try:
    import cupy as cp
except ImportError:
    cp = None


def has_gpu():
    try:  # getDeviceCount raises, rather than returning 0, when there is no driver
        return cp is not None and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class RmFast(Rm):
    def __init__(self, *args, backend="numpy", dtype="float64", shuffle_seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        if backend not in ("numpy", "cupy"):
            raise ValueError(f"backend must be numpy or cupy, got {backend!r}")
        # a CPU run recorded as a GPU arm is worse than a failed job
        if backend == "cupy" and not has_gpu():
            raise RuntimeError("backend='cupy' needs cupy and a visible CUDA device")
        self._backend = backend
        self._dtype = np.dtype(dtype)
        self._shuffle_rng = np.random.default_rng(shuffle_seed)

    def _get_eigen(self, Y):
        if self._backend != "cupy":
            return super()._get_eigen(Y)
        Yg = cp.asarray(Y, dtype=self._dtype)
        L, V = cp.linalg.eigh(Yg)
        L_h = cp.asnumpy(L).astype(np.float64)
        V_h = cp.asnumpy(V).astype(np.float64)
        del Yg, L, V
        cp.get_default_memory_pool().free_all_blocks()
        return L_h, V_h

    def _random_matrix(self, X):
        idx = np.argsort(self._shuffle_rng.random(X.shape), axis=0)
        return np.take_along_axis(X, idx, axis=0)
