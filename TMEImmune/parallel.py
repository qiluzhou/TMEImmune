"""
Parallel helpers: score several cohorts at once, and compute bootstrap confidence intervals.

The scores themselves are array operations, so parallelising inside one score buys little. The work
that does parallelise cleanly is the embarrassingly parallel kind:

    from TMEImmune import parallel
    scores = parallel.score_cohorts({'riaz': {'df': expr1, 'clin': clin1},
                                     'gide': {'df': expr2, 'clin': clin2}}, n_jobs=-1)
    ci = parallel.bootstrap_metric(clin1['resp'], scores['riaz'], n_boot=2000, n_jobs=-1)

Both helpers give the same numbers as the sequential code: cohorts are independent, and every bootstrap
replicate draws its samples from a seed sequence that does not depend on how the work is split.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from TMEImmune import TME_score


# --------------------------------------------------------------------------------------
# cohorts
# --------------------------------------------------------------------------------------
def score_cohorts(cohorts, func = None, n_jobs = -1, backend = 'loky', verbose = 0, **kwargs):
    """
    Score several cohorts in parallel, one worker per cohort.

    cohorts: {name: {'df': expression, 'clin': clinical, ...}}; the inner keys are passed to func,
             and anything in kwargs is added to every call
    func: scoring function (default TME_score.get_all_score)
    n_jobs: worker processes (-1 = all cores). n_jobs=1 runs sequentially, which is what you want
            for debugging, and gives identical results
    backend: 'loky' (processes, default) or 'threading'
    Starting a worker process costs about a second (each one imports the package and reloads the
    packaged data), so this pays off when a cohort takes more than a few seconds; for a handful of tiny
    cohorts n_jobs=1 is faster.
    Output: {name: result}. A cohort that raises is reported as the exception instead of the result,
            so one bad cohort does not lose the rest.
    """
    func = func or TME_score.get_all_score
    names = list(cohorts)

    def run(name):
        args = dict(cohorts[name])
        args.update(kwargs)
        df = args.pop('df')
        clin = args.pop('clin')
        try:
            return func(df, clin, **args)
        except Exception as exc:
            return exc

    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(delayed(run)(n) for n in names)
    return dict(zip(names, results))


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------
METRICS = {
    'auc': roc_auc_score,
    'auprc': average_precision_score,
    'accuracy': lambda y, s: accuracy_score(y, (np.asarray(s) >= 0.5).astype(int)),
    'f1': lambda y, s: f1_score(y, (np.asarray(s) >= 0.5).astype(int), zero_division=0),
}


def _boot_indices(n, y, n_boot, seed, stratified):
    """Bootstrap index matrix (n_boot x n). Stratified resampling keeps the number of responders fixed."""
    rng = np.random.default_rng(seed)
    if not stratified:
        return rng.integers(0, n, size=(n_boot, n))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                 rng.choice(neg, len(neg), replace=True)])
    return out



def _auc_bootstrap_fast(y, x, idx):
    """
    AUC of one score for every bootstrap replicate at once.

    The AUC equals the rank-sum statistic, and a bootstrap replicate only changes how often each sample
    appears, so the whole set of replicates can be evaluated with cumulative sums over the samples sorted
    once by score (ties get mid-ranks, exactly as roc_auc_score does).
    idx: (n_boot x n) matrix of resampled row indices.
    """
    keep = np.isfinite(x)
    order = np.argsort(x[keep], kind='mergesort')
    pos_of = np.full(len(x), -1)
    pos_of[np.where(keep)[0][order]] = np.arange(keep.sum())     # sample -> position in sorted order

    # multiplicity of each sorted position in every replicate
    n_boot, n = idx.shape
    mult = np.zeros((n_boot, keep.sum()))
    rows = np.repeat(np.arange(n_boot), n)
    cols = pos_of[idx.ravel()]
    valid = cols >= 0
    np.add.at(mult, (rows[valid], cols[valid]), 1.0)

    xs = x[keep][order]
    group_start = np.concatenate([[0], np.where(np.diff(xs) != 0)[0] + 1])   # tie groups
    group_mult = np.add.reduceat(mult, group_start, axis=1)
    before = np.cumsum(group_mult, axis=1) - group_mult
    mid_rank = before + (group_mult + 1.0) / 2.0                             # mid-rank of each group

    ys = y[keep][order]
    pos_mult = np.add.reduceat(mult * (ys == 1), group_start, axis=1)
    n_pos = mult[:, ys == 1].sum(axis=1)
    n_neg = mult[:, ys == 0].sum(axis=1)
    rank_sum = (pos_mult * mid_rank).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def bootstrap_metric(y_true, scores, metric = 'auc', n_boot = 1000, n_jobs = -1, seed = 0,
                     stratified = True, alpha = 0.05, batch = 50, backend = 'loky', fast_auc = True):
    """
    Bootstrap confidence intervals for one metric, for every score column, in parallel.

    y_true: 0/1 response indexed by sample (or an array aligned with scores)
    scores: dataframe of scores (samples x scores), or a single Series
    metric: 'auc', 'auprc', 'accuracy', 'f1', or a callable(y_true, score)
    n_boot: bootstrap replicates; batch: replicates per parallel task
    backend: 'loky' (processes, default) or 'threading'
    fast_auc: use the vectorized rank-sum AUC (same values as roc_auc_score, far fewer python calls)
    stratified: resample responders and non-responders separately, so every replicate keeps the
                observed class balance (with 7 responders in 40 patients, plain resampling produces
                replicates with no responders at all, where the metric is undefined)
    Output: dataframe indexed by score with columns estimate, ci_low, ci_high, n_valid.
    """
    metric_fn = METRICS[metric] if isinstance(metric, str) else metric
    if isinstance(scores, pd.Series):
        scores = scores.to_frame()
    if isinstance(y_true, pd.Series):
        common = scores.index.intersection(y_true.index)
        scores, y_true = scores.loc[common], y_true.loc[common]
    y = np.asarray(y_true).astype(int)
    X = scores.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    idx = _boot_indices(len(y), y, n_boot, seed, stratified)

    def run(chunk):
        out = np.full((len(chunk), X.shape[1]), np.nan)
        for i, rows in enumerate(chunk):
            yy = y[rows]
            if len(np.unique(yy)) < 2:
                continue
            for j in range(X.shape[1]):
                col = X[rows, j]
                keep = np.isfinite(col)
                if len(np.unique(yy[keep])) < 2:
                    continue
                out[i, j] = metric_fn(yy[keep], col[keep])
        return out

    if isinstance(metric, str) and metric == 'auc' and fast_auc:
        # vectorized rank-sum AUC: every replicate at once, one score column per task
        cols = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_auc_bootstrap_fast)(y, X[:, j], idx) for j in range(X.shape[1]))
        boot = np.column_stack(cols)
    else:
        chunks = [idx[i:i + batch] for i in range(0, n_boot, batch)]
        parts = Parallel(n_jobs=n_jobs, backend=backend)(delayed(run)(c) for c in chunks)
        boot = np.vstack(parts)

    rows = []
    for j, name in enumerate(scores.columns):
        col = X[:, j]
        keep = np.isfinite(col)
        point = metric_fn(y[keep], col[keep]) if len(np.unique(y[keep])) > 1 else np.nan
        vals = boot[np.isfinite(boot[:, j]), j]
        lo, hi = (np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(vals) else (np.nan, np.nan))
        rows.append({'score': name, 'estimate': point, 'ci_low': lo, 'ci_high': hi, 'n_valid': len(vals)})
    return pd.DataFrame(rows).set_index('score')


def bootstrap_best_score(y_true, scores, metric = 'auc', n_boot = 1000, n_jobs = -1, seed = 0,
                         stratified = True, backend = 'loky'):
    """
    Optimism-corrected estimate for "the best score".

    Picking the highest-AUC score and reporting that AUC on the same patients is optimistic. Here the
    selection is repeated inside every bootstrap replicate and the winner is scored on the patients that
    replicate left out, which estimates how much of the winner's margin is selection.
    Output: dict with the apparent best score and AUC, the optimism-corrected mean, and how often each
            score won.
    """
    metric_fn = METRICS[metric] if isinstance(metric, str) else metric
    if isinstance(y_true, pd.Series):
        common = scores.index.intersection(y_true.index)
        scores, y_true = scores.loc[common], y_true.loc[common]
    y = np.asarray(y_true).astype(int)
    X = scores.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    names = list(scores.columns)
    idx = _boot_indices(len(y), y, n_boot, seed, stratified)

    def run(rows):
        oob = np.setdiff1d(np.arange(len(y)), np.unique(rows))
        if len(oob) < 5 or len(np.unique(y[rows])) < 2 or len(np.unique(y[oob])) < 2:
            return None
        in_scores = [metric_fn(y[rows], X[rows, j]) if np.isfinite(X[rows, j]).all() else np.nan
                     for j in range(X.shape[1])]
        j = int(np.nanargmax(in_scores))
        if not np.isfinite(X[oob, j]).all():
            return None
        return j, metric_fn(y[oob], X[oob, j])

    out = Parallel(n_jobs=n_jobs, backend=backend)(delayed(run)(rows) for rows in idx)
    out = [o for o in out if o is not None]
    wins = pd.Series([names[j] for j, _ in out]).value_counts()
    apparent = {n: metric_fn(y[np.isfinite(X[:, j])], X[np.isfinite(X[:, j]), j]) for j, n in enumerate(names)}
    best = max(apparent, key=apparent.get)
    return {'apparent_best': best, 'apparent_value': apparent[best],
            'corrected_mean': float(np.mean([v for _, v in out])) if out else np.nan,
            'optimism': apparent[best] - (float(np.mean([v for _, v in out])) if out else np.nan),
            'win_counts': wins, 'n_replicates': len(out)}
