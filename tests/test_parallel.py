"""
Check the parallel helpers: several cohorts at once, and bootstrap confidence intervals.
Run from the project root:  python tests/test_parallel.py

Parallel results must equal the sequential ones, and the vectorized AUC bootstrap must equal
scikit-learn's roc_auc_score replicate by replicate.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import parallel, TME_score

warnings.simplefilter("ignore")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def demo_cohort(n_samples=40, seed=0):
    rng = np.random.default_rng(seed)
    example = pd.read_csv(os.path.join(root, "data", "example_gene.csv"), index_col=0, encoding="utf-8-sig")
    genes = [g for g in example.index.dropna().astype(str) if g]
    expr = pd.DataFrame(rng.lognormal(2, 2, (len(genes), n_samples)),
                        index=genes, columns=[f"S{seed}_{i}" for i in range(n_samples)])
    clin = pd.DataFrame({"resp": rng.integers(0, 2, n_samples)}, index=expr.columns)
    return expr, clin


if __name__ == "__main__":
    e1, c1 = demo_cohort(seed=0)
    e2, c2 = demo_cohort(seed=1)
    cohorts = {"cohort_A": {"df": e1, "clin": c1}, "cohort_B": {"df": e2, "clin": c2}}
    scorer = lambda df, clin, **k: TME_score.get_score(df, ["ESTIMATE", "ISTME", "NetBio"], clin, "resp")

    # ---- cohorts in parallel ----
    t0 = time.perf_counter()
    seq = parallel.score_cohorts(cohorts, func=scorer, n_jobs=1)
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    par = parallel.score_cohorts(cohorts, func=scorer, n_jobs=-1)
    t_par = time.perf_counter() - t0
    for name in cohorts:
        assert isinstance(par[name], pd.DataFrame), par[name]
        assert np.allclose(seq[name].to_numpy(float), par[name].to_numpy(float))
    print(f"score_cohorts: identical results, {t_seq:.1f}s sequential vs {t_par:.1f}s on "
          f"{os.cpu_count()} cores")

    # ---- bootstrap ----
    scores = pd.concat([seq["cohort_A"], pd.Series(
        c1["resp"] + np.random.default_rng(5).normal(0, 0.7, len(c1)), name="informative")], axis=1)
    ci_seq = parallel.bootstrap_metric(c1["resp"], scores, n_boot=1000, n_jobs=1, seed=1)
    ci_par = parallel.bootstrap_metric(c1["resp"], scores, n_boot=1000, n_jobs=-1, seed=1)
    assert np.allclose(ci_seq.to_numpy(float), ci_par.to_numpy(float), equal_nan=True)
    print("\nbootstrap 95% CIs (1000 replicates, stratified):")
    print(ci_seq.round(3).to_string())

    # the vectorized AUC equals roc_auc_score on every replicate
    y = c1["resp"].to_numpy()
    x = scores["informative"].to_numpy()
    idx = parallel._boot_indices(len(y), y, 200, 0, True)
    fast = parallel._auc_bootstrap_fast(y, x, idx)
    slow = np.array([roc_auc_score(y[r], x[r]) if len(np.unique(y[r])) > 1 else np.nan for r in idx])
    print(f"vectorized AUC vs roc_auc_score: max |diff| = {np.nanmax(np.abs(fast - slow)):.2e}")
    assert np.nanmax(np.abs(fast - slow)) < 1e-10

    t0 = time.perf_counter()
    parallel.bootstrap_metric(c1["resp"], scores, n_boot=2000, n_jobs=1, seed=1, fast_auc=False)
    t_loop = time.perf_counter() - t0
    t0 = time.perf_counter()
    parallel.bootstrap_metric(c1["resp"], scores, n_boot=2000, n_jobs=1, seed=1, fast_auc=True)
    t_vec = time.perf_counter() - t0
    print(f"bootstrap 2000 replicates: {t_loop:.2f}s looping vs {t_vec:.2f}s vectorized "
          f"({t_loop / max(t_vec, 1e-9):.0f}x)")

    # ---- how much of the best score's margin is selection ----
    best = parallel.bootstrap_best_score(c1["resp"], scores, n_boot=500, n_jobs=-1)
    print(f"\nbest score '{best['apparent_best']}': apparent {best['apparent_value']:.3f}, "
          f"optimism-corrected {best['corrected_mean']:.3f} (optimism {best['optimism']:.3f}, "
          f"{best['n_replicates']} replicates)")
    print("all parallel tests passed")
