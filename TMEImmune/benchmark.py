"""
Runtime and memory benchmarking for the TMEImmune scores.

Used to document the cost of each score and how it scales with the number of samples:

    from TMEImmune import benchmark
    res = benchmark.benchmark_scores(df, clin, response_col='resp', sizes=[50, 100, 200])
    benchmark.plot_benchmark(res, 'benchmark.png')

Every measurement records wall time and peak additional memory (tracemalloc), and each score is timed
twice: 'cold' with all package caches cleared and 'warm' with them in place, which is what a user sees
when several scores are computed one after another in the same session.
"""
import gc
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd

from TMEImmune import data_processing, estimateScore, netbio, SIAscore, ISAFN
from TMEImmune import ISTME as ISM
from TMEImmune import nb_utilities as nbu


def clear_all_caches():
    """Drop every in-memory cache of the package (packaged data files, ssGSEA ranking, NetBio model)."""
    data_processing.clear_data_cache()
    nbu.clear_ssgsea_cache()
    nbu.parse_reactomeExpression_and_immunotherapyResponse.cache_clear()
    netbio.clear_netbio_cache()
    ISAFN._MODEL_CACHE.clear()
    gc.collect()


def timeit(func, *args, repeat = 1, **kwargs):
    """Run func and return (result, seconds, peak MiB). With repeat > 1 the median time is reported."""
    times = []
    tracemalloc.start()
    try:
        for _ in range(repeat):
            gc.collect()
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        peak = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    finally:
        tracemalloc.stop()
    return result, float(np.median(times)), peak


# --------------------------------------------------------------------------------------
# the individual scores, as {name: callable(df, clin)}
# --------------------------------------------------------------------------------------
def score_tasks(response_col = 'resp', df_mut = None, **isafn_kwargs):
    """Callables for the scores that can be benchmarked separately."""
    return {
        'ESTIMATE': lambda df, clin: estimateScore.ESTIMATEscore(df),
        'ISTME': lambda df, clin: ISM.istmeScore(df),
        'SIA': lambda df, clin: SIAscore.sia_score(df),
        'NetBio': lambda df, clin: netbio.get_netbio(df.reset_index(), clin, response_col),
        'ISAFN': lambda df, clin: ISAFN.isafn_score(df.T, clin, response_col, df_mut=df_mut, **isafn_kwargs),
    }


def benchmark_scores(df, clin, response_col = 'resp', sizes = None, scores = None, repeat = 1,
                     warm = True, df_mut = None, seed = 0, verbose = True, **isafn_kwargs):
    """
    Time every score on subsets of a cohort.

    df: gene expression, gene symbols as index and samples as columns
    clin: clinical dataframe indexed by sample, containing response_col
    sizes: sample sizes to benchmark (default: the full cohort only). Sizes above the cohort size are
           obtained by sampling samples with replacement, which is how you get a scaling curve from a
           small cohort -- the timings are honest, the scores are not meaningful.
    scores: subset of the score names (default: all)
    repeat: repetitions per measurement (the median is reported)
    warm: also time each score a second time with the caches warm
    Output: a dataframe with one row per (score, size, cache state) and columns seconds / peak_MiB.
    """
    rng = np.random.default_rng(seed)
    samples = [s for s in df.columns if s in clin.index]
    if not samples:
        raise ValueError("no overlapping samples between df columns and clin index")
    sizes = sizes or [len(samples)]
    tasks = score_tasks(response_col, df_mut, **isafn_kwargs)
    if scores is not None:
        tasks = {k: v for k, v in tasks.items() if k in scores}

    rows = []
    for n in sizes:
        replace = n > len(samples)
        picked = list(rng.choice(samples, n, replace=replace))
        sub = df[picked]
        sub_clin = clin.loc[picked]
        if replace:                                    # de-duplicate the labels of resampled columns
            new_names = [f"{s}_{i}" for i, s in enumerate(picked)]
            sub = sub.copy(); sub.columns = new_names
            sub_clin = sub_clin.copy(); sub_clin.index = new_names
        for name, fn in tasks.items():
            for state in (['cold', 'warm'] if warm else ['cold']):
                if state == 'cold':
                    clear_all_caches()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, secs, peak = timeit(fn, sub, sub_clin, repeat=repeat)
                    rows.append({'score': name, 'n_samples': n, 'cache': state,
                                 'seconds': secs, 'peak_MiB': peak})
                except Exception as exc:               # a score may fail on a subset (e.g. one class left)
                    rows.append({'score': name, 'n_samples': n, 'cache': state,
                                 'seconds': np.nan, 'peak_MiB': np.nan, 'error': repr(exc)[:200]})
                if verbose:
                    r = rows[-1]
                    print(f"{name:10s} n={n:<6d} {state:4s} "
                          f"{r['seconds'] if np.isfinite(r.get('seconds', np.nan)) else 'failed':>10}"
                          f"{'' if 'error' not in r else '  ' + r['error']}")
    return pd.DataFrame(rows)


def benchmark_ssgsea(df, geneset = None, repeat = 1, reference = True):
    """Compare the fast ssGSEA with the original per-sample implementation on the same data."""
    rows = []
    nbu.clear_ssgsea_cache()
    _, secs, peak = timeit(nbu.ssgsea, df, geneset=geneset, verbose=False, repeat=repeat)
    rows.append({'implementation': 'fast', 'n_samples': df.shape[1], 'seconds': secs, 'peak_MiB': peak})
    if reference:
        _, secs, peak = timeit(nbu.ssgsea_reference, df, geneset=geneset, repeat=repeat)
        rows.append({'implementation': 'reference', 'n_samples': df.shape[1], 'seconds': secs, 'peak_MiB': peak})
    out = pd.DataFrame(rows)
    if reference:
        out['speedup'] = out['seconds'].iloc[1] / out['seconds']
    return out


def compare_implementations(df, clin, response_col = 'resp', scores = ('ESTIMATE', 'ISTME', 'NetBio'),
                            repeat = 1, df_mut = None, **isafn_kwargs):
    """
    Before / after table for the optimisation work: the original per-sample ssGSEA with no caching,
    versus the closed-form ssGSEA, first with cold caches and then with the caches warm.
    Output: a dataframe with one row per score and setting, plus the speedup over the original.
    """
    tasks = score_tasks(response_col, df_mut, **isafn_kwargs)
    tasks = {k: v for k, v in tasks.items() if k in scores}
    rows = []
    original = nbu.ssgsea
    for setting in ['original', 'fast (cold)', 'fast (warm)']:
        nbu.ssgsea = nbu.ssgsea_reference if setting == 'original' else original
        for name, fn in tasks.items():
            if setting != 'fast (warm)':
                clear_all_caches()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, secs, peak = timeit(fn, df, clin, repeat=repeat)
            rows.append({'score': name, 'setting': setting, 'seconds': secs, 'peak_MiB': peak})
    nbu.ssgsea = original
    out = pd.DataFrame(rows).pivot(index='score', columns='setting', values='seconds')
    out['speedup (cold)'] = out['original'] / out['fast (cold)']
    out['speedup (warm)'] = out['original'] / out['fast (warm)']
    return out


def plot_benchmark(result, path = None, value = 'seconds'):
    """Plot runtime (or peak memory) against the number of samples, one line per score."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    data = result[result['cache'] == 'cold'] if 'cache' in result else result
    for name, grp in data.groupby('score'):
        grp = grp.sort_values('n_samples')
        ax.plot(grp['n_samples'], grp[value], marker='o', lw=2, label=name)
    ax.set_xlabel('number of samples', fontweight='bold')
    ax.set_ylabel('seconds' if value == 'seconds' else 'peak memory (MiB)', fontweight='bold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('TMEImmune score cost', fontweight='bold')
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight')
    return fig
