"""
Run ISAFN on the Riaz et al. cohort (data/example_riaz.xlsx) and report its performance.
Run from the project root:  python tests/riaz_example.py

Sheets: 'riaz' = expression (genes x samples, TPM), 'Sheet1' = clinical, 'Sheet2' = mutation calls (one row per variant).
"""
import os
import warnings
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from lifelines.utils import concordance_index

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import ISAFN

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
expr = pd.read_excel(xlsx, "riaz").set_index("gene").T                 # samples x genes
clin = pd.read_excel(xlsx, "Sheet1").dropna(subset=["ID"]).set_index("ID")
clin["resp"] = (clin["response"] == "R").astype(int)
maf = pd.read_excel(xlsx, "Sheet2")

# long-format mutation calls -> samples x genes mutation-count matrix
mut = ISAFN.mutation_to_matrix(maf, sample_col="ID", gene_col="Hugo Symbol")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = ISAFN.isafn_score(expr, clin, "resp", df_mut=mut, drug_col="drug", met_col="Metastasis", cancer="melanoma")
res = res.join(clin[["resp", "delta", "Time to Death\n(weeks)"]])


def metrics(df, prob, pred):
    y, yhat = df["resp"], df[pred].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": len(df), "n_R": int(y.sum()), "AUC": roc_auc_score(y, df[prob]),
            "accuracy": accuracy_score(y, yhat), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "F1": f1_score(y, yhat, zero_division=0),
            "C-index": concordance_index(df["Time to Death\n(weeks)"], df[prob], df["delta"])}


with_mut = res.dropna(subset=["isafn_fusion_prob"])
table = pd.DataFrame({
    "expr model (all samples)": metrics(res, "isafn_expr_prob", "isafn_expr_pred"),
    "expr model (samples with mutation)": metrics(with_mut, "isafn_expr_prob", "isafn_expr_pred"),
    "fusion model (samples with mutation)": metrics(with_mut, "isafn_fusion_prob", "isafn_fusion_pred"),
}).T
print(table.round(4).to_string())
