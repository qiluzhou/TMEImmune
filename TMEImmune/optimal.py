import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
import warnings


def assign_type(score, upper_p, lower_p):
    """
    Split samples by score quantiles.
    One cutoff (upper_p == lower_p): H (>= cutoff) and L (< cutoff).
    Two cutoffs: H (>= upper quantile), L (<= lower quantile) and M (in between).
    """
    if upper_p == lower_p:
        q = score.quantile(upper_p)
        def get_type(value):
            return 'H' if value >= q else 'L'
    else:
        upper_q = score.quantile(upper_p)
        lower_q = score.quantile(lower_p)
        def get_type(value):
            if value >= upper_q:
                return 'H'
            elif value <= lower_q:
                return 'L'
            else:
                return 'M'
    return score.apply(get_type)


def _top_scores(metric_dict, top_n):
    """Score names ordered by metric (best first), limited to top_n (None = all)."""
    ordered = sorted(metric_dict, key=lambda k: metric_dict[k], reverse=True)
    return ordered if top_n is None else ordered[:top_n]


def _binary_response(df, response):
    df1 = df[~df[response].isna()].copy()
    all_strings = df1[response].map(lambda x: isinstance(x, str)).all()
    if all_strings:
        resp = df1[response].str.upper()
        if resp.isin(["R", "NR"]).all():
            df1["binary_resp"] = (resp == "R").astype(int)
        else:
            raise ValueError("Unsupported response type")
    elif df1[response].isin([0, 1]).all():
        df1["binary_resp"] = df1[response].astype(int)
    else:
        raise ValueError("Unsupported response type")
    return df1


def _optimal_ICI(df, response, score_name, name = None, roc = True, top_n = 5):
    """
    Find the optimal score with the highest AUC for ICI response
    df: pandas dataframe with row index as sample ID, columns containing scores, ICI response ([R, NR] or [0, 1])
    response: column name of ICI response
    score_name: list of names of the score columns in df, or a string of name for only one score
    name: custom title name for figure
    roc: whether to draw the ROC curves
    top_n: number of best scores (highest AUC) drawn in the ROC figure; None draws all scores
    output: a figure showing the ROC curves of the top scores (None if roc = False), and a dictionary with all
            scores and the corresponding AUC
    """
    df1 = _binary_response(df, response)
    if isinstance(score_name, str):
        score_name = [score_name]

    auc_dict, roc_dict, n_dict = {}, {}, {}
    for score in score_name:
        s = pd.to_numeric(df1[score], errors = 'coerce').astype(float)
        if np.isinf(s).any():
            warnings.warn(f"Warning: {score} column contains infinity values. These rows will be removed.", UserWarning)
        if s.isna().any():
            warnings.warn(f"Warning: {score} column contains NaN values. These rows will be removed.", UserWarning)
        keep = np.isfinite(s)
        y = df1.loc[keep, 'binary_resp']
        if y.nunique() < 2:
            warnings.warn(f"Warning: {score} has only one response class after removing missing values; skipped.", UserWarning)
            continue
        fpr, tpr, _ = roc_curve(y, s[keep])
        auc_dict[score] = auc(fpr, tpr)
        roc_dict[score] = (fpr, tpr)
        n_dict[score] = int(keep.sum())

    if not auc_dict:
        raise ValueError("No score could be evaluated")

    fig = None
    if roc:
        fig = plt.figure(figsize=(8, 6), dpi = 100)
        plt.margins(x=0.05, y=0.05)
        for score in _top_scores(auc_dict, top_n):
            fpr, tpr = roc_dict[score]
            plt.plot(fpr, tpr, lw=4, label=f'{score} ({auc_dict[score]:.4f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlim([0.0, 1.03])
        plt.ylim([0.0, 1.1])
        plt.xlabel("False Positive Rates", fontweight = "bold", fontsize = 14)
        plt.ylabel("True Positive Rates", fontweight = "bold", fontsize = 14)
        plt.xticks(fontweight="bold", fontsize = 14)
        plt.yticks(fontweight="bold", fontsize = 14)
        title = "ROC Curves" if name is None else f"ROC Curves for {name}"
        if top_n is not None and len(auc_dict) > top_n:
            title += f" (top {top_n})"
        plt.title(title, fontweight = "bold", fontsize = 16)
        plt.legend(loc="lower right", title = "AUC",
                   title_fontproperties={'weight': 'bold', 'size': 14},
                   prop = {'size':12, 'weight' : "bold"}, frameon = False)

    best_auc = max(auc_dict.values())
    optimal_score = [k for k, v in auc_dict.items() if v == best_auc]
    print(f"The optimal score for ICI response prediction: {optimal_score} with AUC = {round(best_auc, 4)}")

    return fig, auc_dict, n_dict


def _logrank_HL(df1, score, delta, time, upper_p, lower_p):
    """Log-rank p-value comparing the high (H) and low (L) score groups."""
    groups = assign_type(df1[score], upper_p, lower_p)
    H, L = df1[groups == 'H'], df1[groups == 'L']
    if len(H) == 0 or len(L) == 0:
        return np.nan
    res = logrank_test(H[time], L[time], event_observed_A=H[delta], event_observed_B=L[delta])
    return res.p_value


def _optimal_survival(df, delta, time, score_name, clinical_factors = None, upper_p = 0.75, lower_p = 0.25, name = None,
                     km_curves = True, palette = ["#547AC0", "#898988", "#F6C957"], top_n = 5):
    """
    Compare score performance to survival prognosis using the c-index (Cox-PH model when clinical factors are
    given). Each score is divided into groups by quantiles, and users can choose to add other clinical factors
    into the model to adjust for confounding variables.
    delta: column name for survival status. 1 as event (death) and 0 otherwise
    time: column name for survival time, should be float type
    score_name: list of names of the score columns in df, or a string of name for only one score
    clinical factors: list of column names of the other clinical factors added to the model
    upper_p, lower_p: the probabilities at which the upper and lower quantiles are calculated, by default 0.75 and 0.25
    name: custom title name for the figure
    km_curves: whether to display the Kaplan-Meier survival curves of the optimal score
    palette: colors of the H, L and M groups in the K-M survival curves
    top_n: number of best scores (highest c-index) listed in the figure; None lists all scores
    output: the K-M figure (None if km_curves = False), and a dictionary with all scores and their c-index
    """
    df1 = df[~df[delta].isna() & ~df[time].isna()].copy()
    df1[time] = pd.to_numeric(df1[time], errors='coerce')
    df1[delta] = pd.to_numeric(df1[delta], errors='coerce')
    if not df1[delta].isin([1, 0]).all():
        raise ValueError("delta must be a column containing only 0 or 1")

    if isinstance(score_name, str):
        score_name = [score_name]
    if clinical_factors is not None:
        if isinstance(clinical_factors, str):
            clinical_factors = [clinical_factors]
        elif not isinstance(clinical_factors, list):
            raise TypeError("Invalid clinical factor input")

    c_dict, p_dict, n_dict = {}, {}, {}
    for score in score_name:
        cols = [delta, time, score] + (clinical_factors or [])
        data = df1[cols].copy()
        data[score] = pd.to_numeric(data[score], errors='coerce').astype(float)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 2 or data[delta].sum() == 0:
            warnings.warn(f"Warning: not enough samples/events to evaluate {score}; skipped.", UserWarning)
            continue
        if clinical_factors is None:
            c_ind = concordance_index(data[time], data[score], data[delta])
        else:
            cph = CoxPHFitter()
            cph.fit(data, duration_col = time, event_col = delta)
            predicted_risk = cph.predict_partial_hazard(data)
            # higher hazard means shorter survival, so negate for the concordance index
            c_ind = concordance_index(data[time], -predicted_risk, data[delta])
        c_dict[score] = c_ind
        p_dict[score] = _logrank_HL(data, score, delta, time, upper_p, lower_p)
        n_dict[score] = len(data)

    if not c_dict:
        raise ValueError("No score could be evaluated")

    best_c = max(c_dict.values())
    optimal_score = [k for k, v in c_dict.items() if v == best_c]
    print(f"The optimal score for survival prognosis: {optimal_score} with c-index = {round(best_c, 4)}")

    fig = None
    if km_curves:
        fig, axs = plt.subplots(len(optimal_score), 1, figsize=(8, 6 * len(optimal_score)), dpi = 100, squeeze = False)
        axs = axs[:, 0]
        top = _top_scores(c_dict, top_n)
        for i, score in enumerate(optimal_score):
            data = df1[[delta, time, score]].copy()
            data[score] = pd.to_numeric(data[score], errors='coerce')
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            data['score_type'] = assign_type(data[score], upper_p, lower_p)
            groups = [('H', palette[0]), ('M', palette[2]), ('L', palette[1])] if upper_p != lower_p else \
                     [('H', palette[0]), ('L', palette[1])]
            for g, color in groups:
                sub = data[data['score_type'] == g]
                if len(sub) == 0:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(durations=sub[time], event_observed=sub[delta], label=g)
                kmf.plot_survival_function(show_censors = True, ci_show = False, linewidth=4, color = color, ax = axs[i])

            axs[i].margins(x=0.05, y=0.05)
            axs[i].text(1.03, 1.0, f"H vs L: P = {p_dict[score]:.4f}", transform=axs[i].transAxes,
                        verticalalignment='top', fontweight = 'bold', fontsize = 13)
            header = "C-index" if top_n is None or len(c_dict) <= top_n else f"C-index (top {top_n})"
            text_str = header + "\n" + "\n".join(f"{k}: {c_dict[k]:.4f}" for k in top)
            bbox_props = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="none", linewidth=1)
            axs[i].text(1.03, 0.9, text_str, fontsize=12, verticalalignment="top", bbox=bbox_props,
                        multialignment="left", linespacing=1.4, transform=axs[i].transAxes, fontweight = 'bold')

            title = f"K-M Curves for {score}" if name is None else f"K-M Curves for {score} in {name}"
            axs[i].set_title(title, fontweight = 'bold', fontsize = 16)
            axs[i].set_xlabel(time, fontweight = 'bold', fontsize = 14)
            axs[i].set_ylabel("Survival Rate", fontweight = 'bold', fontsize = 14)
            for lab in axs[i].get_xticklabels() + axs[i].get_yticklabels():
                lab.set_fontweight('bold')
                lab.set_fontsize(14)
            axs[i].legend(loc="lower left", title="Group",
                          title_fontproperties={'weight': 'bold', 'size': 12},
                          prop={'size': 10, 'weight': "bold"})
        fig.tight_layout()

    return fig, c_dict, p_dict, n_dict


def optimal_ICI(df, response, score_name, name = None, roc = True, top_n = 5):
    """
    Find the optimal score with the highest AUC for ICI response
    df: pandas dataframe with row index as sample ID, columns containing scores, ICI response ([R, NR] or [0, 1])
    response: column name of ICI response
    score_name: list of names of the score columns in df, or a string of name for only one score
    name: custom title name for figure
    roc: whether to draw the ROC curves
    top_n: number of best scores (highest AUC) drawn in the ROC figure; None draws all scores
    output: the ROC figure (None if roc = False) and a dictionary with all scores and their AUC
    """
    fig, auc_dict, _ = _optimal_ICI(df, response, score_name, name, roc, top_n)
    return fig, auc_dict


def optimal_survival(df, delta, time, score_name, clinical_factors = None, upper_p = 0.75, lower_p = 0.25, name = None,
                     km_curves = True, palette = ["#547AC0", "#898988", "#F6C957"], top_n = 5):
    """
    Compare score performance to survival prognosis using the c-index (Cox-PH model when clinical factors are
    given), and draw the K-M curves of the optimal score split by score quantiles.
    delta: column name for survival status. 1 as event (death) and 0 otherwise
    time: column name for survival time, should be float type
    score_name: list of names of the score columns in df, or a string of name for only one score
    clinical factors: list of column names of the other clinical factors added to the model
    upper_p, lower_p: quantile probabilities for the H / L groups, by default 0.75 and 0.25 (equal values give two groups)
    name: custom title name for the figure
    km_curves: whether to display the Kaplan-Meier survival curves of the optimal score
    palette: colors of the H, L and M groups in the K-M survival curves
    top_n: number of best scores (highest c-index) listed in the figure; None lists all scores
    output: the K-M figure (None if km_curves = False) and a dictionary with all scores and their c-index
    """
    fig, c_dict, _, _ = _optimal_survival(df, delta, time, score_name, clinical_factors, upper_p, lower_p, name,
                                          km_curves, palette, top_n)
    return fig, c_dict


def performance_table(auc_dict = None, c_dict = None, logrank_p = None, n_ici = None, n_surv = None):
    """
    Summarize the performance of all scores in one table, sorted by AUC (or c-index when only survival is given).
    Columns: AUC and its rank for ICI response; c-index, H vs L log-rank p-value and rank for survival.
    """
    parts = []
    if auc_dict:
        t = pd.DataFrame({'AUC': pd.Series(auc_dict)})
        if n_ici:
            t['n (ICI)'] = pd.Series(n_ici)
        t['AUC rank'] = t['AUC'].rank(ascending = False, method = 'min').astype(int)
        parts.append(t)
    if c_dict:
        t = pd.DataFrame({'C-index': pd.Series(c_dict)})
        if logrank_p:
            t['log-rank P (H vs L)'] = pd.Series(logrank_p)
        if n_surv:
            t['n (survival)'] = pd.Series(n_surv)
        t['C-index rank'] = t['C-index'].rank(ascending = False, method = 'min').astype(int)
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, axis = 1)
    table = table.sort_values('AUC' if auc_dict else 'C-index', ascending = False)
    table.index.name = 'score'
    return table


def get_performance(df_score, metric, score_name, surv_col = None, surv_p = None, ICI_col = None, df_clin = None,
                    clinical_factors = None, show_fig = True, name = None, top_n = 5, return_table = False):
    """
    Compare score performance of ICI therapy or/and survival prognosis
    df_score: pandas dataframe containing scores with sample ID as row index
    metric: choose from ["survival", "ICI"]. Enter the full list if the user wants to compare both.
    surv_col: column names of the survival comparison, should be a list of [status, time]
    surv_p: quantile probability for score segmentation in K-M survival curves
    ICI_col: column name of the response column
    score_name: list of names of the score columns in df, or a string of name for only one score
    df_clin: if the columns of survival or response are not in df_score, provide them as a column with sample ID as row index in the pandas dataframe
    clinical_factors: additional clinical columns adjusted for in the survival (Cox-PH) model
    show_fig: whether to show the ROC curves/K-M survival curves, default to True
    top_n: number of best scores shown in the figures (default 5); None shows all. The table always has all scores.
    return_table: also return the performance table of all scores
    output: a list of (figure, dictionary) per metric with all scores and their AUCs/c-indices.
            A performance table of all scores is printed; with return_table = True the output is (list, table).
    """
    if isinstance(metric, str):
        metric = [metric]
    elif not isinstance(metric, list):
        raise TypeError("Invalid format, must be a list of metrics or a string of a single metric")

    df = df_score.copy()
    if df_clin is not None:
        extra = df_clin.drop(columns = [c for c in df_clin.columns if c in df.columns])
        df = df.merge(extra, left_index = True, right_index = True, how = "inner")

    outcomes = []
    auc_dict = c_dict = p_dict = n_ici = n_surv = None
    for m in metric:
        if m == "survival":
            if not isinstance(surv_col, list):
                raise TypeError("The input survival columns must be in a list")
            if len(surv_col) != 2:
                raise ValueError("The input survival columns must only have status and time")
            if df[surv_col[0]].dropna().isin([1, 0]).all():
                delta, time = surv_col[0], surv_col[1]
            else:
                delta, time = surv_col[1], surv_col[0]

            if surv_p is None:
                upper_p, lower_p = 0.75, 0.25
            elif isinstance(surv_p, float):
                upper_p = lower_p = surv_p
            elif isinstance(surv_p, list):
                upper_p, lower_p = max(surv_p), min(surv_p)
            else:
                raise TypeError("Invalid input format of quantile probabilities")
            fig, c_dict, p_dict, n_surv = _optimal_survival(df, delta, time, score_name, clinical_factors = clinical_factors,
                                                            upper_p = upper_p, lower_p = lower_p, name = name,
                                                            km_curves = show_fig, top_n = top_n)
            outcomes.append((fig, c_dict))
        elif m == "ICI":
            if not isinstance(ICI_col, str):
                raise TypeError("The input ICI_col must be a string")
            fig, auc_dict, n_ici = _optimal_ICI(df, ICI_col, score_name, name = name, roc = show_fig, top_n = top_n)
            outcomes.append((fig, auc_dict))
        else:
            raise ValueError("Invalid metric, must be survival or ICI")

    table = performance_table(auc_dict, c_dict, p_dict, n_ici, n_surv)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250, 'display.float_format', '{:.4f}'.format):
        print("\nScore performance (all scores):")
        print(table)

    if return_table:
        return outcomes, table
    return outcomes


# ---------------------------------------------------------------------------------------------
# cross-cohort batch correction: does a correction keep the biology and remove the cohort?
# ---------------------------------------------------------------------------------------------

def batch_separability(expr, batch, n_components = 10, seed = 0):
    """
    How much cohort structure is left in an expression matrix -- the batch effect that survived a
    correction. Lower is better.

    expr: expression matrix, genes as rows and samples as columns
    batch: cohort label per sample, aligned to expr.columns
    Output: dict with
        silhouette: silhouette score of the cohort labels in the space of the leading principal
                    components. 1 means the cohorts sit in completely separate clusters, 0 means they
                    overlap entirely, and negative values mean samples sit closer to another cohort
                    than to their own
        n_genes, n_components: what the score was computed from

    This fits nothing: PCA and the silhouette are both descriptive, so the number cannot be inflated by
    a classifier that happened to train well.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    batch = pd.Series(batch, index=expr.columns).astype(str)
    X = expr.T.apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.notna().all(axis=0) & (X.std(axis=0) > 0)]
    if X.shape[1] < 2 or batch.nunique() < 2:
        return {'silhouette': float('nan'), 'n_genes': int(X.shape[1]), 'n_components': 0}
    Z = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    k = int(min(n_components, Z.shape[1], Z.shape[0] - 1))
    pcs = PCA(n_components=k, random_state=seed).fit_transform(Z)
    return {'silhouette': float(silhouette_score(pcs, batch.to_numpy())),
            'n_genes': int(X.shape[1]), 'n_components': k}


def compare_batch_correction(expression, clinical, score_fn, methods = ('none', 'combat',
                             'minmax_global', 'minmax_cohort'), response_col = 'resp',
                             source_col = 'source', protect = ('resp',), verbose = True):
    """
    Compare cross-cohort corrections on the two things that matter at once: how much of the response
    signal survives, and how much of the cohort effect is gone.

    A correction that scores well on signal alone can be cheating -- it may have amplified a cohort
    difference that happens to line up with response, which is exactly the failure mode of pooled ICI
    data. Reading the two columns together is what makes the comparison honest.

    expression: merged expression matrix, genes as rows and samples as columns
    clinical: merged clinical table indexed by sample, with the response and cohort columns
    score_fn: callable (expression, clinical) -> dataframe of scores indexed by sample. Usually a
              lambda wrapping TME_score.get_all_score with the arguments for this dataset
    methods: the corrections to compare (see data_processing.correct_batch)
    Output: (summary dataframe, detail dict). The summary has one row per method and score with
            pooled AUC and the batch-separability numbers.

    Nothing here fits a predictive model to your data: every score is computed by the existing scoring
    functions and only compared across corrections.
    """
    from sklearn.metrics import roc_auc_score
    from TMEImmune import data_processing as dp

    clinical = clinical.reindex(expression.columns)
    y = pd.to_numeric(clinical[response_col], errors='coerce')
    batch = clinical[source_col].astype(str)
    covar = clinical[[c for c in protect if c in clinical.columns]] if protect else None

    rows, detail = [], {}
    for method in methods:
        try:
            corrected, rep = dp.correct_batch(expression, batch, method=method, covariates=covar,
                                              verbose=False)
        except Exception as exc:
            if verbose:
                print(f"---------- {method}: not available ({type(exc).__name__}: "
                      f"{str(exc)[:80]}) ----------")
            detail[method] = {'error': f"{type(exc).__name__}: {exc}"}
            continue
        sep = batch_separability(corrected, batch)
        scores = score_fn(corrected, clinical)
        detail[method] = {'report': rep, 'separability': sep, 'scores': scores}

        for col in scores.columns:
            if str(col).endswith('_pred'):
                continue
            s = pd.to_numeric(scores[col], errors='coerce').reindex(clinical.index)
            ok = s.notna() & y.notna()
            if ok.sum() < 10 or y[ok].nunique() < 2:
                continue
            pooled = roc_auc_score(y[ok], s[ok])
            # per-cohort AUC as well: a pooled AUC can be inflated purely by a cohort difference
            per_cohort = []
            for label in batch.unique():
                m = ok & (batch == label)
                if m.sum() >= 10 and y[m].nunique() > 1:
                    per_cohort.append(roc_auc_score(y[m], s[m]))
            rows.append({'method': method, 'score': col, 'auc': pooled,
                         'auc_directional': max(pooled, 1 - pooled),
                         'batch_silhouette': sep['silhouette'],
                         'auc_within_cohort': float(np.mean(per_cohort)) if per_cohort else np.nan,
                         'n_cohorts_scored': len(per_cohort), 'n': int(ok.sum())})

    summary = pd.DataFrame(rows)
    if verbose and len(summary):
        print("\n---------- batch correction: signal kept vs cohort effect removed ----------")
        print("(auc: response signal, higher is better | batch_silhouette: cohort structure "
              "remaining, lower is better)")
        per_method = summary.groupby('method').agg(
            mean_auc=('auc_directional', 'mean'), best_auc=('auc_directional', 'max'),
            batch_silhouette=('batch_silhouette', 'first'),
            auc_within_cohort=('auc_within_cohort', 'mean'), n_scores=('score', 'count'))
        print(per_method.round(4).to_string())
    return summary, detail


# ---------------------------------------------------------------------------------------------
# decision curve analysis: is acting on a score better than treating everyone or no one?
# ---------------------------------------------------------------------------------------------

def _calibrate(score, y, cv = None, seed = 0):
    """
    Turn one score onto a probability scale with a single-variable logistic fit.

    cv: None fits on all the data, which is the usual decision-curve setup but optimistic, because the
        same patients set the calibration and then judge it. An integer instead returns out-of-fold
        predictions from that many folds, which removes most of that optimism.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    x = np.asarray(score, dtype=float)
    # standardise first, and fit without a penalty: sklearn's default L2 shrinks the coefficient, so an
    # unstandardised score with a narrow range (ISAFN's probabilities span about 0.1) would be shrunk
    # towards a constant risk and look useless for reasons that are purely about its units
    sd = x.std()
    X = ((x - x.mean()) / (sd if sd > 0 else 1.0)).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    model = LogisticRegression(max_iter=1000, penalty=None)
    if cv:
        folds = int(min(cv, np.bincount(y).min()))
        if folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            return cross_val_predict(model, X, y, cv=splitter, method='predict_proba')[:, 1]
    return model.fit(X, y).predict_proba(X)[:, 1]


def net_benefit(risk, y, thresholds):
    """
    Net benefit of treating everyone whose predicted risk reaches each threshold.

        NB = TP/n - (FP/n) * p/(1-p)

    The second term prices a false positive in units of true positives: at a threshold of 0.2 a
    clinician is saying they would treat four patients needlessly to catch one who benefits.
    """
    risk = np.asarray(risk, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    out = []
    for p in thresholds:
        treat = risk >= p
        tp = int(np.sum(treat & (y == 1)))
        fp = int(np.sum(treat & (y == 0)))
        out.append(tp / n - (fp / n) * (p / (1 - p)) if p < 1 else np.nan)
    return np.asarray(out)


def decision_curve(df_score, response, score_name = None, thresholds = None, cv = None,
                   seed = 0, verbose = True):
    """
    Decision curve analysis: the clinical value of acting on each score, across the range of
    thresholds a clinician might plausibly use.

    AUC says how well a score ranks patients; it says nothing about whether using it beats the two
    strategies that need no score at all -- treat everyone, or treat no one. That is what this answers.

    df_score: dataframe of scores, samples as index (e.g. the output of TME_score.get_all_score)
    response: binary response per sample -- a Series indexed like df_score, or the name of a column
    score_name: which scores to include (default: every numeric column except *_pred)
    thresholds: threshold probabilities to evaluate (default 0.01 to 0.99)
    cv: out-of-fold calibration with this many folds; None (default) calibrates on all the data
    Output: (curves, summary). curves is a dataframe of net benefit indexed by threshold, with a
            column per score plus 'Treat all' and 'Treat none'. summary ranks the scores by
            area_above_default -- how much net benefit they add over the better of treating everyone
            and treating no one -- and gives the best margin, where it occurs, and the range of
            thresholds over which the score is worth using at all.

    Every score is put on a probability scale by a single-variable logistic fit, since net benefit is
    only defined against a risk threshold. With cv=None that fit sees the same patients it is judged
    on, so the curves are optimistic in absolute terms; the comparison between scores is still fair,
    because every score is calibrated the same way.
    """
    y = df_score[response] if isinstance(response, str) and response in df_score.columns else response
    y = pd.Series(y).reindex(df_score.index)
    y = pd.to_numeric(y, errors='coerce')
    if y.dropna().nunique() != 2:
        raise ValueError("response must be binary (0/1) and vary across the samples given")

    names = score_name or [c for c in df_score.columns
                           if not str(c).endswith('_pred') and c != response]
    thresholds = np.asarray(thresholds if thresholds is not None else np.arange(0.01, 1.00, 0.01))

    curves = pd.DataFrame(index=pd.Index(thresholds, name='threshold'))
    rows = []
    prevalence = float(y.dropna().mean())
    curves['Treat all'] = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    curves['Treat none'] = 0.0

    for name in names:
        s = pd.to_numeric(df_score[name], errors='coerce')
        ok = s.notna() & y.notna()
        if ok.sum() < 10 or y[ok].nunique() < 2:
            continue
        risk = _calibrate(s[ok], y[ok], cv=cv, seed=seed)
        nb = net_benefit(risk, y[ok].to_numpy(), thresholds)
        curves[name] = nb
        # what matters is the margin over the better of the two strategies that need no score at all;
        # at a very low threshold every score simply treats everyone, so raw net benefit there is just
        # the response rate and tells you nothing
        default = np.maximum(curves['Treat all'].to_numpy(), 0.0)
        advantage = nb - default
        better = advantage > 0
        rng = (float(thresholds[better].min()), float(thresholds[better].max())) if better.any() else None
        rows.append({'score': name,
                     'area_above_default': float(np.trapezoid(np.maximum(advantage, 0), thresholds)),
                     'best_advantage': float(np.nanmax(advantage)),
                     'best_threshold': float(thresholds[int(np.nanargmax(advantage))]),
                     'useful_from': None if rng is None else rng[0],
                     'useful_to': None if rng is None else rng[1],
                     'n_thresholds_better_than_default': int(better.sum()),
                     'area_under_net_benefit': float(np.trapezoid(nb, thresholds)), 'n': int(ok.sum())})

    summary = pd.DataFrame(rows).sort_values('area_above_default', ascending=False)
    if verbose and len(summary):
        print(f"---------- decision curve analysis: {len(summary)} scores, {int(y.notna().sum())} "
              f"samples, response rate {prevalence:.1%} "
              f"({'out-of-fold' if cv else 'in-sample'} calibration) ----------")
        show = summary.head(10)[['score', 'area_above_default', 'best_advantage', 'best_threshold',
                                 'useful_from', 'useful_to', 'n']].copy()
        for c in ('area_above_default', 'best_advantage', 'best_threshold', 'useful_from', 'useful_to'):
            show[c] = show[c].astype(float).round(4)
        print(show.to_string(index=False))
    return curves, summary


def plot_decision_curve(curves, summary = None, top_n = 5, path = None, name = None, ylim = None):
    """
    Draw the decision curves for the most useful scores, against treat-all and treat-none.

    curves, summary: the output of decision_curve
    top_n: how many scores to draw, ranked by area under the net-benefit curve
    """
    scores = [c for c in curves.columns if c not in ('Treat all', 'Treat none')]
    if summary is not None and len(summary):
        scores = [s for s in summary['score'].tolist() if s in scores][:top_n]
    else:
        scores = scores[:top_n]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(curves.index, curves['Treat none'], color='0.55', lw=1.4, ls=':', label='Treat none')
    ax.plot(curves.index, curves['Treat all'], color='0.25', lw=1.4, ls='--', label='Treat all')
    for s in scores:
        ax.plot(curves.index, curves[s], lw=1.8, label=s)

    lo = min(0.0, float(np.nanmin(curves[scores].to_numpy()))) if scores else -0.05
    ax.set_ylim(ylim or (max(lo, -0.15), float(np.nanmax(curves[scores + ['Treat all']].to_numpy())) * 1.15))
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(name or 'Decision curve analysis')
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        print(f"---------- decision curve saved to {path} ----------")
    return fig


# ---------------------------------------------------------------------------------------------
# redundancy: does a score say anything the others do not?
# ---------------------------------------------------------------------------------------------

def score_correlation(df_score, score_name = None, method = 'spearman', cluster = True,
                      threshold = 0.9, verbose = True):
    """
    How much the scores duplicate each other, and which ones form near-identical groups.

    Rank correlation is the right measure here because the scores live on completely different scales
    and only their ordering of patients is comparable.

    df_score: dataframe of scores, samples as index
    method: 'spearman' (default), 'kendall' or 'pearson'
    cluster: order the matrix by hierarchical clustering on 1 - |correlation|
    threshold: correlation above which two scores are called near-duplicates
    Output: (correlation dataframe, dict with the clustered order, the flat clusters, and the list of
            near-duplicate pairs)
    """
    names = score_name or [c for c in df_score.columns if not str(c).endswith('_pred')]
    X = df_score[names].apply(pd.to_numeric, errors='coerce')
    X = X.loc[:, X.notna().sum() >= 10]
    X = X.loc[:, X.std(numeric_only=True) > 0]
    corr = X.corr(method=method)

    info = {'method': method, 'n_scores': corr.shape[0], 'n_samples': int(len(X))}
    if cluster and corr.shape[0] > 2:
        from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
        from scipy.spatial.distance import squareform
        d = 1 - corr.abs().to_numpy()
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2                                  # linkage needs an exactly symmetric matrix
        link = linkage(squareform(d, checks=False), method='average')
        leaves = leaves_list(link)
        groups = fcluster(link, t=1 - threshold, criterion='distance')
        order = [corr.index[i] for i in leaves]             # fcluster is indexed like the input matrix
        info['clusters'] = {corr.index[i]: int(groups[i]) for i in leaves}
        corr = corr.loc[order, order]
        info['order'] = order

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append((cols[i], cols[j], float(r)))
    info['near_duplicates'] = sorted(pairs, key=lambda t: -abs(t[2]))

    if verbose:
        print(f"---------- {method} correlation across {info['n_scores']} scores "
              f"({info['n_samples']} samples) ----------")
        if info['near_duplicates']:
            print(f"---------- {len(info['near_duplicates'])} near-duplicate pairs "
                  f"(|r| >= {threshold}) ----------")
            for a, b, r in info['near_duplicates'][:10]:
                print(f"     {a} ~ {b}: r = {r:+.3f}")
        else:
            print(f"---------- no pair reaches |r| >= {threshold} ----------")
        if 'clusters' in info:
            n_groups = len(set(info['clusters'].values()))
            print(f"---------- {n_groups} groups at |r| >= {threshold} ----------")
    return corr, info


def discordant_pairs(df_score, response, reference, comparator = None, verbose = True):
    """
    Among the patient pairs the existing scores already order, how often does the reference score
    disagree -- and when it does, which one is right?

    Every pair of one responder and one non-responder is a question both scores answer: which of these
    two responded? The fraction each gets right is its AUC. Restricting to the pairs where they
    disagree turns that into a paired comparison: of the pairs only one score gets right, how many go
    to the reference. A score that is a relabelling of another has almost no disagreeing pairs; a score
    with its own information has many, and wins more than half of them.

    df_score: dataframe of scores, samples as index
    response: binary response per sample, or the name of a column in df_score
    reference: the score being argued for, e.g. 'isafn_expr_prob'
    comparator: which scores to compare against (default: all the others)
    Output: dataframe with, per comparator, the two AUCs, how many pairs the two scores disagree on,
            how many of those the reference wins, and a two-sided sign-test p-value.

    Nothing is fitted: this is a rank comparison on the scores as they are.
    """
    from scipy.stats import binomtest

    y = df_score[response] if isinstance(response, str) and response in df_score.columns else response
    y = pd.to_numeric(pd.Series(y).reindex(df_score.index), errors='coerce')
    if reference not in df_score.columns:
        raise ValueError(f"reference score '{reference}' is not a column of df_score")
    names = comparator or [c for c in df_score.columns
                           if c != reference and not str(c).endswith('_pred') and c != response]
    if isinstance(names, str):
        names = [names]

    rows = []
    for name in names:
        a = pd.to_numeric(df_score[reference], errors='coerce')
        b = pd.to_numeric(df_score[name], errors='coerce')
        ok = a.notna() & b.notna() & y.notna()
        if ok.sum() < 10 or y[ok].nunique() < 2:
            continue
        av, bv, yv = a[ok].to_numpy(), b[ok].to_numpy(), y[ok].to_numpy().astype(int)
        pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
        if len(pos) == 0 or len(neg) == 0:
            continue
        # every responder x non-responder pair; 1 when the score puts the responder higher, 0.5 on a tie
        da = np.sign(av[pos][:, None] - av[neg][None, :])
        db = np.sign(bv[pos][:, None] - bv[neg][None, :])
        ca, cb = (da + 1) / 2, (db + 1) / 2                  # 1 correct, 0.5 tie, 0 wrong
        # a pair counts as a disagreement whenever one score does better on it than the other, which
        # includes a score breaking a tie the other could not; the sign test uses only the clear-cut
        # pairs, where one is right and the other is wrong
        ref_better = int(np.sum(ca > cb))
        cmp_better = int(np.sum(ca < cb))
        disagree = ref_better + cmp_better
        ref_only = int(np.sum((ca == 1) & (cb == 0)))
        cmp_only = int(np.sum((ca == 0) & (cb == 1)))
        clear = ref_only + cmp_only
        p = binomtest(ref_only, clear).pvalue if clear else np.nan
        rows.append({'comparator': name, 'auc_reference': float(ca.mean()),
                     'auc_comparator': float(cb.mean()),
                     'auc_difference': float(ca.mean() - cb.mean()),
                     'n_pairs': int(ca.size), 'n_disagree': disagree,
                     'fraction_disagree': disagree / ca.size if ca.size else np.nan,
                     'reference_wins': ref_better, 'comparator_wins': cmp_better,
                     'reference_win_rate': ref_better / disagree if disagree else np.nan,
                     'n_clear_pairs': clear, 'reference_clear_wins': ref_only,
                     'n_tied_pairs': int(np.sum((ca == 0.5) | (cb == 0.5))),
                     'sign_test_p': float(p) if clear else np.nan})

    out = pd.DataFrame(rows).sort_values('auc_difference', ascending=False)
    if verbose and len(out):
        print(f"---------- {reference} vs {len(out)} other scores, on the pairs where they "
              f"disagree ----------")
        show = out[['comparator', 'auc_reference', 'auc_comparator', 'auc_difference',
                    'fraction_disagree', 'reference_win_rate', 'sign_test_p']].copy()
        print(show.round(4).to_string(index=False))
    return out


def disagreement_cases(df_score, response, reference, comparator, n = 20, verbose = True):
    """
    The individual patients two scores disagree about most, and what actually happened to them.

    The pair analysis says how often a score is right when it dissents; this says who those patients
    are, which is what makes the difference concrete in a paper or a clinic.

    df_score: dataframe of scores, samples as index
    response: binary response per sample, or the name of a column in df_score
    reference, comparator: the two score columns to contrast
    n: how many patients to return from each end of the disagreement
    Output: (cases dataframe, summary dict). cases holds the samples with the largest gap between the
            two scores' rank percentiles, with the direction of the disagreement and the observed
            response; summary gives the response rate in each direction.
    """
    y = df_score[response] if isinstance(response, str) and response in df_score.columns else response
    y = pd.to_numeric(pd.Series(y).reindex(df_score.index), errors='coerce')
    a = pd.to_numeric(df_score[reference], errors='coerce')
    b = pd.to_numeric(df_score[comparator], errors='coerce')
    ok = a.notna() & b.notna() & y.notna()
    ra, rb = a[ok].rank(pct=True), b[ok].rank(pct=True)
    gap = ra - rb

    cases = pd.DataFrame({'reference_pct': ra, 'comparator_pct': rb, 'gap': gap,
                          'response': y[ok].astype(int)})
    cases['direction'] = np.where(cases['gap'] > 0, f'{reference} higher', f'{comparator} higher')
    cases = cases.reindex(cases['gap'].abs().sort_values(ascending=False).index)

    ref_high = cases[cases['gap'] > 0].head(n)
    cmp_high = cases[cases['gap'] < 0].head(n)
    summary = {
        'n_compared': int(ok.sum()), 'overall_response_rate': float(y[ok].mean()),
        f'{reference}_higher_n': int(len(ref_high)),
        f'{reference}_higher_response_rate': float(ref_high['response'].mean()) if len(ref_high) else np.nan,
        f'{comparator}_higher_n': int(len(cmp_high)),
        f'{comparator}_higher_response_rate': float(cmp_high['response'].mean()) if len(cmp_high) else np.nan,
        'median_absolute_gap': float(cases['gap'].abs().median()),
    }
    if verbose:
        print(f"---------- {reference} vs {comparator}: the {n} patients each ranks most differently "
              f"----------")
        print(f"     overall response rate {summary['overall_response_rate']:.1%} over "
              f"{summary['n_compared']} patients")
        print(f"     where {reference} is higher: {summary[f'{reference}_higher_response_rate']:.1%} "
              f"responded (n={summary[f'{reference}_higher_n']})")
        print(f"     where {comparator} is higher: "
              f"{summary[f'{comparator}_higher_response_rate']:.1%} responded "
              f"(n={summary[f'{comparator}_higher_n']})")
    return pd.concat([ref_high, cmp_high]), summary
