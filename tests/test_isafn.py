"""
Smoke test for ISAFN.isafn_score and TME_score.get_all_score on pseudo data.
Run from the project root (needs torch, gseapy and the other TMEImmune dependencies):

    python tests/test_isafn.py
"""
import json
import os
import warnings
import numpy as np
import pandas as pd

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import ISAFN, TME_score, data_processing, optimal


def make_pseudo_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "TMEImmune", "data", "isafn", "features_output.json")) as f:
        feat = json.load(f)
    example = pd.read_csv(os.path.join(root, "data", "example_gene.csv"), index_col=0, encoding="utf-8-sig")
    mmr = ['MLH1', 'MSH2', 'MSH3', 'MSH6', 'PMS2']
    genes = sorted(set(example.index.dropna().astype(str)) | set(sum(feat.values(), [])) | set(mmr))
    samples = [f"S{i:03d}" for i in range(n)]

    # gene expression: samples x genes, log2(TPM + 1)-like values
    base = rng.uniform(1, 9, len(genes))
    expr = pd.DataFrame(np.clip(base + rng.normal(0, 1.2, (n, len(genes))), 0, None), index=samples, columns=genes)

    # mutation: samples x genes, mutation counts; last 5 samples have no mutation data
    mut_genes = sorted(set(feat['selected_mut'] + feat['selected_mut_male'] + feat['selected_mut_female']) | set(mmr))
    mut = pd.DataFrame((rng.random((n, len(mut_genes))) < 0.03) * rng.integers(1, 4, (n, len(mut_genes))),
                       index=samples, columns=mut_genes)
    mut.loc[samples[:3], 'MLH1'] = 1
    mut = mut.drop(index=samples[-5:])

    # clinical: ICI response, sex, drug, metastasis status, cohort
    clin = pd.DataFrame(index=samples)
    clin['resp'] = rng.integers(0, 2, n)
    clin['sex'] = rng.choice(['M', 'F'], n, p=[0.6, 0.4])
    clin.loc[samples[10:12], 'sex'] = np.nan
    clin['drug'] = rng.choice(['Pembrolizumab', 'Nivolumab', 'Ipilimumab + Nivolumab'], n)
    clin['metastasis'] = rng.choice(['yes', 'no'], n)
    clin['source'] = np.where(np.arange(n) < 35, 'cohortA', 'cohortB')
    clin['delta'] = rng.integers(0, 2, n)
    clin['time'] = np.round(rng.exponential(24, n) + 1, 2)
    return expr, mut, clin


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    expr, mut, clin = make_pseudo_data()

    out = ISAFN.isafn_score(expr, clin, 'resp', df_mut=mut, gender_col='sex', drug_col='drug',
                            met_col='metastasis', cancer='melanoma', sex_output=True)
    print(out.head(), "\n", out.describe())
    assert list(out.columns) == ['isafn_expr_prob', 'isafn_expr_pred', 'isafn_fusion_prob', 'isafn_fusion_pred']
    assert list(out.index) == list(clin.index)
    assert out['isafn_expr_prob'].between(0, 1).all()
    assert out['isafn_fusion_prob'].isna().sum() == 5

    merged = ISAFN.isafn_score(expr, clin, df_mut=mut, gender_col='sex', drug_col='drug',
                               met_col='metastasis', sex_output=False)
    unknown_sex = clin.index[clin['sex'].isna()]
    assert np.allclose(out.loc[unknown_sex, 'isafn_expr_prob'], merged.loc[unknown_sex, 'isafn_expr_prob'])

    # sex imputation from chrY / XIST expression when sex is not recorded
    imp = ISAFN.isafn_score(expr, clin, df_mut=mut, drug_col='drug', met_col='metastasis', impute_sex=True)
    assert imp['isafn_expr_prob'].notna().all()

    expr_only = ISAFN.isafn_score(expr, clin)
    assert list(expr_only.columns) == ['isafn_expr_prob', 'isafn_expr_pred']

    allscore = TME_score.get_all_score(expr.T, clin, response_col='resp', source_col='source', df_mut=mut,
                                       gender_col='sex', drug_col='drug', met_col='metastasis')
    print(allscore.head())

    # input harmonisation: the same cohort in different units
    tpm = 2 ** expr - 1                                              # samples x genes, TPM scale
    rng2 = np.random.default_rng(7)
    fpkm = tpm.div(tpm.sum(axis=1), axis=0) * rng2.uniform(4e5, 3e6, len(tpm))[:, None]
    assert data_processing.detect_expression_type(expr.T)['type'] == 'log'
    assert data_processing.detect_expression_type(tpm.T)['type'] in ('tpm_or_cpm', 'fpkm_or_tpm')
    assert data_processing.detect_expression_type(fpkm.T)['type'] == 'fpkm_or_tpm'
    zscored = expr.sub(expr.mean(axis=0), axis=1).div(expr.std(axis=0).replace(0, 1), axis=1)
    det_z = data_processing.detect_expression_type(zscored.T)
    assert det_z['type'] == 'zscore' and not det_z['convertible']

    # FPKM -> TPM is exact, so FPKM and TPM give the same ISAFN scores
    out_tpm = ISAFN.isafn_score(tpm, clin, 'resp', df_mut=mut, gender_col='sex', drug_col='drug', met_col='metastasis')
    out_fpkm = ISAFN.isafn_score(fpkm, clin, 'resp', df_mut=mut, gender_col='sex', drug_col='drug', met_col='metastasis')
    assert np.allclose(out_tpm.astype(float).fillna(-1).values, out_fpkm.astype(float).fillna(-1).values, atol=1e-6)
    assert out_tpm.attrs['input_report']['detected']['type'] in ('tpm_or_cpm', 'fpkm_or_tpm')

    # log-scale input is left alone, so it reproduces the run above
    out_log = ISAFN.isafn_score(expr, clin, 'resp', df_mut=mut, gender_col='sex', drug_col='drug', met_col='metastasis')
    assert np.allclose(out_log.astype(float).fillna(-1).values, out.astype(float).fillna(-1).values, atol=1e-6)
    assert np.allclose(data_processing.normalization(df=tpm.T).values, expr.T.values, atol=1e-6)

    # compare all scores with optimal.py (ICI response AUC + survival c-index)
    score_cols = [c for c in allscore.columns if not c.endswith('_pred')]
    ((roc_fig, auc), (km_fig, cindex)), table = optimal.get_performance(
        allscore, metric=['ICI', 'survival'], score_name=score_cols, ICI_col='resp',
        surv_col=['delta', 'time'], surv_p=[0.25, 0.75], df_clin=clin[['resp', 'delta', 'time']],
        top_n=5, return_table=True)
    assert set(auc) == set(score_cols) and set(cindex) == set(score_cols)
    assert len(table) == len(score_cols)                                   # table has every score
    assert len(roc_fig.axes[0].get_legend().get_texts()) == 5              # figure shows the top 5
    print("all ISAFN tests passed")
