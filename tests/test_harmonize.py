"""
Input harmonisation: detect what an expression matrix is and convert it to log2(TPM+1).
Run from the project root:  python tests/test_harmonize.py

One cohort (Riaz, data/example_riaz.xlsx, TPM) is re-expressed as raw counts, FPKM, log2 TPM and
z-scored values. The same matrix in different units should be detected correctly, converted back to the
same scale, and give the same ISAFN scores; z-scored data cannot be converted and must say so.
"""
import os
import warnings

import numpy as np
import pandas as pd

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import data_processing as dp
from TMEImmune import ISAFN

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cohort_formats(seed=0):
    """The Riaz cohort in five units. Gene lengths are random stand-ins: only the round trip matters."""
    xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
    tpm = pd.read_excel(xlsx, "riaz").set_index("gene")                      # genes x samples
    clin = pd.read_excel(xlsx, "Sheet1").dropna(subset=["ID"]).set_index("ID")
    clin["resp"] = (clin["response"] == "R").astype(int)
    samples = [c for c in tpm.columns if "Pre_" in c and c in clin.index]
    tpm, clin = tpm[samples], clin.loc[samples]

    rng = np.random.default_rng(seed)
    lengths = pd.Series(rng.integers(500, 12000, len(tpm)), index=tpm.index)
    counts = (tpm.div(1e6).mul(lengths / 1000, axis=0) * 3e7).round()        # TPM -> counts, 30M reads
    fpkm = tpm.div(tpm.sum(0), axis=1) * rng.uniform(4e5, 3e6, tpm.shape[1])  # arbitrary per-sample scale
    log2tpm = np.log2(tpm + 1)
    zscored = log2tpm.sub(log2tpm.mean(1), axis=0).div(log2tpm.std(1).replace(0, 1), axis=0)
    return tpm, counts, fpkm, log2tpm, zscored, clin, lengths


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    tpm, counts, fpkm, log2tpm, zscored, clin, lengths = cohort_formats()
    print(f"cohort: {tpm.shape[0]} genes x {tpm.shape[1]} samples\n")

    # ---- 1. detection ----
    expected = {"TPM (subset of genes)": ("fpkm_or_tpm", "tpm_or_cpm"), "counts": ("counts",),
                "FPKM": ("fpkm_or_tpm",), "log2 TPM": ("log",), "z-scored": ("zscore",),
                "CPM": ("tpm_or_cpm",)}
    cpm = counts.div(counts.sum(0), axis=1) * 1e6
    for name, mat in [("TPM (subset of genes)", tpm), ("counts", counts), ("FPKM", fpkm),
                      ("log2 TPM", log2tpm), ("z-scored", zscored), ("CPM", cpm)]:
        det = dp.detect_expression_type(mat)
        print(f"{name:22s} -> {det['type']:12s} log={str(det['log']):5s} "
              f"convertible={str(det['convertible']):5s} | {det['message'][:64]}")
        assert det["type"] in expected[name], (name, det["type"])

    # ---- 2. conversion ----
    print("\nconversions to log2(TPM+1):")
    ref, _ = dp.to_log2tpm(tpm, verbose=False)
    conv_fpkm, rep = dp.to_log2tpm(fpkm, verbose=False)
    print(f"  FPKM  : {rep['steps']} -> identical to TPM: "
          f"{np.allclose(conv_fpkm.to_numpy(), ref.to_numpy(), atol=1e-8)}")
    assert np.allclose(conv_fpkm.to_numpy(), ref.to_numpy(), atol=1e-8)

    # counts -> TPM renormalises over the genes present, so it is compared with the harmonised TPM
    # (the raw TPM of this file was computed over all genes and sums to less than 1e6 here)
    conv_counts, rep = dp.to_log2tpm(counts, gene_length=lengths, verbose=False)
    print(f"  counts + lengths: max |diff| vs harmonised TPM = "
          f"{np.abs(conv_counts.to_numpy() - ref.loc[conv_counts.index].to_numpy()).max():.4f}, "
          f"vs the file's own log2 TPM = "
          f"{np.abs(conv_counts.to_numpy() - log2tpm.loc[conv_counts.index].to_numpy()).max():.4f} "
          f"(a per-sample offset, since the file holds TPM for a subset of the genes)")
    assert np.abs(conv_counts.to_numpy() - ref.loc[conv_counts.index].to_numpy()).max() < 0.5

    conv_cpm, rep = dp.to_log2tpm(counts, verbose=False)
    print(f"  counts, no lengths: {rep['steps'][0]}")

    conv_log, rep = dp.to_log2tpm(log2tpm, verbose=False)
    print(f"  log2 TPM: {rep['steps']} -> unchanged: {np.allclose(conv_log.to_numpy(), log2tpm.to_numpy())}")
    assert np.allclose(conv_log.to_numpy(), log2tpm.to_numpy())

    conv_z, rep = dp.to_log2tpm(zscored, verbose=False)
    print(f"  z-scored: converted={rep['converted']}, unchanged={np.allclose(conv_z.to_numpy(), zscored.to_numpy())}")
    assert not rep["converted"]

    # ---- 3. ISAFN through the harmonisation ----
    from sklearn.metrics import roc_auc_score

    def isafn(mat, **kw):
        out = ISAFN.isafn_score(mat.T, clin, drug_col="drug", met_col="Metastasis", impute_sex=True, **kw)
        return out, roc_auc_score(clin.loc[out.index, "resp"], out["isafn_expr_prob"])

    print("\nISAFN on the same cohort in different units:")
    out_tpm, auc_tpm = isafn(tpm)
    out_fpkm, auc_fpkm = isafn(fpkm)
    out_counts, auc_counts = isafn(counts, gene_length=lengths)
    out_log, auc_log = isafn(log2tpm)
    for name, out, auc in [("TPM", out_tpm, auc_tpm), ("FPKM", out_fpkm, auc_fpkm),
                           ("counts + lengths", out_counts, auc_counts), ("log2 TPM", out_log, auc_log)]:
        det = out.attrs["input_report"]["detected"]["type"]
        print(f"  {name:18s} detected={det:12s} AUC={auc:.4f}")
    assert np.allclose(out_tpm["isafn_expr_prob"], out_fpkm["isafn_expr_prob"], atol=1e-6)
    assert np.abs(out_counts["isafn_expr_prob"] - out_tpm["isafn_expr_prob"]).max() < 0.05

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_z, auc_z = isafn(zscored)
        warned = [str(w.message) for w in caught if "z-scored" in str(w.message)]
    print(f"  z-scored          AUC={auc_z:.4f}, warned={bool(warned)}")
    assert warned
    print("\nall harmonisation tests passed")
