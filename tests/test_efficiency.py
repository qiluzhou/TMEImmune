"""
Check the fast ssGSEA against the original implementation and measure the caching / speed work.
Run from the project root:  python tests/test_efficiency.py

Prints three things worth keeping for the paper:
  1. agreement between the closed-form ssGSEA and the original per-sample version,
  2. the before / after table (original, fast with cold caches, fast with warm caches),
  3. runtime and peak memory of every score against the number of samples.
"""
import os
import warnings

import numpy as np
import pandas as pd

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import benchmark, data_processing
from TMEImmune import nb_utilities as nbu

warnings.simplefilter("ignore")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(root, "data", "benchmark")
os.makedirs(out_dir, exist_ok=True)


def synthetic(n_genes=15000, n_samples=60, seed=0):
    """Synthetic cohort built on real gene symbols, so NetBio finds genes in common with its
    training cohort; the expression values themselves are random."""
    rng = np.random.default_rng(seed)
    example = pd.read_csv(os.path.join(root, "data", "example_gene.csv"), index_col=0, encoding="utf-8-sig")
    real = [g for g in example.index.dropna().astype(str) if g]
    genes = list(dict.fromkeys(real + [f"FILLER{i}" for i in range(max(0, n_genes - len(real)))]))[:n_genes]
    expr = pd.DataFrame(rng.lognormal(2, 2, (len(genes), n_samples)),
                        index=genes, columns=[f"S{i}" for i in range(n_samples)])
    expr.iloc[-2000:] = 0.0                                  # zero-expression genes: one large tie group
    clin = pd.DataFrame({"resp": rng.integers(0, 2, n_samples)}, index=expr.columns)
    return expr, clin


if __name__ == "__main__":
    expr, clin = synthetic()
    rng = np.random.default_rng(1)
    gs = {f"sig{i}": list(rng.choice(expr.index, rng.integers(10, 200), replace=False)) for i in range(200)}

    # ---- 1. the fast ssGSEA reproduces the original ----
    fast = nbu.ssgsea(expr, geneset=gs, score="ESTIMATE", verbose=False)
    ref = nbu.ssgsea_reference(expr, geneset=gs, score="ESTIMATE")
    diff = np.abs(fast[ref.columns].to_numpy() - ref.to_numpy())
    corr = np.corrcoef(fast[ref.columns].to_numpy().ravel(), ref.to_numpy().ravel())[0, 1]
    print(f"ssGSEA agreement: max |diff| = {diff.max():.2e} "
          f"(relative {diff.max() / np.abs(ref.to_numpy()).max():.1e}), correlation = {corr:.8f}")
    assert corr > 0.999

    # tie-free data: the two implementations are identical up to floating point
    cont = pd.DataFrame(np.random.default_rng(2).normal(size=(3000, 20)),
                        index=[f"G{i}" for i in range(3000)], columns=[f"S{i}" for i in range(20)])
    gs_small = {k: [g for g in v if g in cont.index] for k, v in list(gs.items())[:30]}
    a = nbu.ssgsea(cont, geneset=gs_small, score="ESTIMATE", verbose=False)
    b = nbu.ssgsea_reference(cont, geneset=gs_small, score="ESTIMATE")
    print(f"without ties: max |diff| = {np.abs(a[b.columns].to_numpy() - b.to_numpy()).max():.2e}")
    assert np.abs(a[b.columns].to_numpy() - b.to_numpy()).max() < 1e-8

    # the fast version does not depend on the order of the genes, the original does
    perm = np.random.default_rng(3).permutation(expr.index)
    p1 = nbu.ssgsea(expr, geneset=gs, score="ESTIMATE", verbose=False)
    p2 = nbu.ssgsea(expr.loc[perm], geneset=gs, score="ESTIMATE", verbose=False)
    r1 = nbu.ssgsea_reference(expr, geneset=gs, score="ESTIMATE")
    r2 = nbu.ssgsea_reference(expr.loc[perm], geneset=gs, score="ESTIMATE")
    print(f"gene-order invariance: fast = {np.abs(p1 - p2[p1.columns]).to_numpy().max():.2e}, "
          f"original = {np.abs(r1 - r2[r1.columns]).to_numpy().max():.2e}")
    assert np.abs(p1 - p2[p1.columns]).to_numpy().max() < 1e-6

    # ---- 2. caches ----
    data_processing.clear_data_cache()
    _, cold, _ = benchmark.timeit(data_processing.load_data, "c2.all.v7.2.symbols.gmt")
    _, warm, _ = benchmark.timeit(data_processing.load_data, "c2.all.v7.2.symbols.gmt")
    print(f"Reactome GMT parse: {cold:.2f}s -> {warm * 1e3:.2f}ms when cached")

    # ---- 3. before / after, and scaling ----
    table = benchmark.compare_implementations(expr, clin, "resp", scores=("ESTIMATE", "ISTME", "NetBio"))
    print("\noriginal vs fast (seconds):")
    print(table.round(3).to_string())
    table.to_csv(os.path.join(out_dir, "optimisation_table.csv"))

    scaling = benchmark.benchmark_scores(expr, clin, response_col="resp", sizes=[20, 60, 120],
                                         scores=("ESTIMATE", "ISTME", "SIA", "NetBio"), verbose=False)
    print("\nruntime and memory by cohort size:")
    print(scaling.round(3).to_string(index=False))
    scaling.to_csv(os.path.join(out_dir, "scaling.csv"), index=False)
    benchmark.plot_benchmark(scaling, os.path.join(out_dir, "scaling.png"))
    print(f"\nwrote {out_dir}/optimisation_table.csv, scaling.csv, scaling.png")
    print("all efficiency tests passed")
