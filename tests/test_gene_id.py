"""
Gene identifier harmonisation: detect Ensembl / Entrez / RefSeq / symbol indices and map them onto
current HGNC symbols. Run from the project root:  python tests/test_gene_id.py

Needs the packaged annotation table (TMEImmune/data/gene_annotation.npz); build it once with
tools/build_gene_annotation.py.
"""
import os
import warnings

import numpy as np
import pandas as pd

import sys

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import gene_id, data_processing as dp, ISAFN

warnings.simplefilter("ignore")
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    if not gene_id.annotation_available():
        raise SystemExit("no gene annotation table; build it with tools/build_gene_annotation.py")
    ann = gene_id.load_annotation()
    genes = ann["genes"]
    print(f"annotation: {ann['source']} | {len(genes)} symbols, "
          f"{(genes['entrez'] != '').sum()} Entrez ids, {(genes['ensembl'] != '').sum()} Ensembl ids, "
          f"{int(np.isfinite(genes['length']).sum())} gene lengths, {len(ann['alias'])} alias symbols")

    xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
    tpm = pd.read_excel(xlsx, "riaz").set_index("gene")
    clin = pd.read_excel(xlsx, "Sheet1").dropna(subset=["ID"]).set_index("ID")
    clin["resp"] = (clin["response"] == "R").astype(int)
    samples = [c for c in tpm.columns if "Pre_" in c and c in clin.index]
    tpm, clin = tpm[samples], clin.loc[samples]

    # ---- 1. detection ----
    to_entrez = genes[genes["entrez"] != ""].drop_duplicates("symbol").set_index("symbol")["entrez"]
    in_table = [g for g in tpm.index if g in to_entrez.index]
    cases = {
        "symbol": pd.Index(in_table[:3000]),
        "entrez": pd.Index(to_entrez.loc[in_table[:3000]].tolist()),
        "ensembl": pd.Index([f"ENSG{i:011d}.{i % 9 + 1}" for i in range(3000)]),
        "refseq": pd.Index([f"NM_{i:06d}" for i in range(3000)]),
    }
    for expected, idx in cases.items():
        det = gene_id.detect_id_type(idx)
        print(f"  {expected:8s} -> detected {det['type']:8s} e.g. {det['examples'][:2]}")
        assert det["type"] == expected

    # ---- 2. Entrez ids convert back to the original symbols ----
    sub = tpm.loc[in_table]
    entrez_mat = sub.copy()
    entrez_mat.index = to_entrez.loc[in_table].to_numpy()
    conv, rep = gene_id.to_symbol(entrez_mat, verbose=False)
    print(f"\nEntrez -> symbol: {rep['mapped']}/{rep['n_input']} mapped, "
          f"{rep['duplicates_merged']} duplicate symbols merged")
    common = conv.index.intersection(sub.index)
    assert len(common) > 0.95 * sub.index.nunique()
    assert np.allclose(conv.loc[common].to_numpy(),
                       sub[~sub.index.duplicated()].loc[common].to_numpy())

    # ---- 3. duplicates are merged the way you ask ----
    dup = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3),
                       index=[in_table[0], in_table[0], in_table[1], in_table[1]], columns=list("xyz"))
    for how, expect in [("sum", 3.0), ("mean", 1.5), ("max", 3.0), ("first", 0.0)]:
        got, _ = gene_id.to_symbol(dup, id_type="symbol", aggregate=how, verbose=False)
        assert got.loc[in_table[0], "x"] == expect, (how, got.loc[in_table[0], "x"])
    print("duplicate symbols: sum / mean / max / first all behave as documented")

    # ---- 4. symbols are never dropped just because the table is older than the data ----
    with_new = tpm.copy()
    with_new.index = list(tpm.index[:-1]) + ["A_BRAND_NEW_SYMBOL"]
    kept, rep = gene_id.to_symbol(with_new, verbose=False)
    assert "A_BRAND_NEW_SYMBOL" in kept.index
    print("unknown symbols are kept (they are usually newer than the annotation)")

    # ---- 5. same scores whether the matrix carries symbols or Entrez ids ----
    from sklearn.metrics import roc_auc_score
    by_symbol = ISAFN.isafn_score(tpm.T, clin, drug_col="drug", met_col="Metastasis", impute_sex=True)
    by_entrez = ISAFN.isafn_score(entrez_mat.T, clin, drug_col="drug", met_col="Metastasis", impute_sex=True)
    auc_s = roc_auc_score(clin.loc[by_symbol.index, "resp"], by_symbol["isafn_expr_prob"])
    auc_e = roc_auc_score(clin.loc[by_entrez.index, "resp"], by_entrez["isafn_expr_prob"])
    print(f"\nISAFN: symbols AUC {auc_s:.4f} ({tpm.shape[0]} genes), "
          f"Entrez ids AUC {auc_e:.4f} ({entrez_mat.shape[0]} genes), "
          f"max |difference| {np.abs(by_symbol['isafn_expr_prob'] - by_entrez['isafn_expr_prob']).max():.1e}")
    assert np.abs(by_symbol["isafn_expr_prob"] - by_entrez["isafn_expr_prob"]).max() < 0.01

    # ---- 6. identifiers and units in one call ----
    out, rep = dp.harmonize(df=entrez_mat, verbose=False)
    print(f"\nharmonize: identifiers {rep['identifiers']['id_type']} -> symbol "
          f"({rep['identifiers']['mapped']} mapped), values {rep['detected']['type']} -> {rep['steps']}")
    assert rep["identifiers"]["id_type"] == "entrez"
    print("\nall gene id tests passed")
