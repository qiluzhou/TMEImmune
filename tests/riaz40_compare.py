"""
Run the merged ISAFN models (real torch) on the 40 Riaz samples used in training evaluation and print
the same summary as the training script, to check that the package reproduces the training results.
Run from the project root:  python tests/riaz40_compare.py
"""
import os
import warnings
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import ISAFN

SAMPLES = """Pt4_Pre_E9021023.6 Pt82_Pre_AD823914.8 Pt103_Pre_AE134058.2 Pt65_Pre_AD793919.6 Pt84_Pre_AD486532.5
Pt46_Pre_AD467096.6 Pt72_Pre_AD793922.5 Pt17_Pre_E9047563.6 Pt27_Pre_AD453873.5 Pt36_Pre_AD467095.6
Pt34_Pre_AD466985.6 Pt26_Pre_AD467789.6 Pt9_Pre_E9021024.6 Pt5_Pre_E9021022.6 Pt94_Pre_AD732850.6
Pt30_Pre_AD497503.5 Pt37_Pre_AD502452.5 Pt38_Pre_E9200719.6 Pt52_Pre_AD506075.6 Pt101_Pre_AD486328.5
Pt59_Pre_AD823915.5 Pt31_Pre_AD453872.5 Pt77_Pre_AD733591.7 Pt29_Pre_AD497504.5 Pt2_Pre_AD101150.6
Pt98_Pre_AD733586.8 Pt85_Pre_AD486329.5 Pt78_Pre_AD467018.5 Pt106_Pre_AD502250.5 Pt62_Pre_AD608303.5
Pt39_Pre_AD485899.5 Pt67_Pre_AD506074.6 Pt1_Pre_AD101148.6 Pt10_Pre_E9047565.6 Pt24_Pre_AD436687.5
Pt48_Pre_E9047561.7 Pt18_Pre_E9024732.6 Pt47_Pre_AD506073.6 Pt90_Pre_AD467873.6 Pt89_Pre_AE070951.5""".split()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
expr = pd.read_excel(xlsx, "riaz").set_index("gene").T.loc[SAMPLES]
clin = pd.read_excel(xlsx, "Sheet1").dropna(subset=["ID"]).set_index("ID").loc[SAMPLES]
clin["resp"] = (clin["response"] == "R").astype(int)
clin["met_mstage"] = (clin["Mstage"] - 1).astype(int)          # M1A / M1B / M1C -> 0 / 1 / 2
mut = ISAFN.mutation_to_matrix(pd.read_excel(xlsx, "Sheet2"), sample_col="ID", gene_col="Hugo Symbol")
mut = mut.reindex(SAMPLES).fillna(0)                            # 5 samples have no mutation calls -> all zero

for met_col in ["Metastasis", "met_mstage"]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ISAFN.isafn_score(expr, clin, "resp", df_mut=mut, drug_col="drug", met_col=met_col, sex_output=False)
    y = clin["resp"]
    print(f"\n===== metastasis encoded from {met_col} =====")
    for model in ["expr", "fusion"]:
        p, yhat = res[f"isafn_{model}_prob"], res[f"isafn_{model}_pred"].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        print(f"[{model}] predicted prob for pos and neg {p[y == 1].mean():.4f} {p[y == 0].mean():.4f}")
        print(f"[{model}] Confusion Matrix -> TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"[{model}] AUC: {roc_auc_score(y, p):.4f}")
