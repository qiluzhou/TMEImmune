"""
Multi-cohort harmonisation: three cohorts that disagree about everything, run one at a time, in pairs
and all together.   python tests/test_cohorts.py [path/to/pseudo_data]

The cohorts differ in gene identifiers (symbols vs Ensembl), expression scale (log2(TPM+1), TPM, raw
counts), response encoding (Responder/Non-responder, CR/PR/SD/PD, 1/0), clinical column names, cancer
type (melanoma vs urothelial), and coverage (cohort B is missing five genes, cohort C has no mutation
data). Every one of those has to be resolved before the scores can be compared.

What is asserted:
  1. each cohort's identifiers, expression scale and clinical encodings are detected correctly
  2. the merged matrix keeps every gene (union), leaving NaN only where a cohort did not measure it
  3. all seven cohort combinations score end to end without raising
  4. a score that cannot be computed from this input is reported, not silently wrong
  5. corrections are compared on signal kept *and* cohort effect removed
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

# run against the working copy in this repository, not a TMEImmune installed in site-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import data_processing as dp, gene_id, TME_score, optimal

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pseudo_cohorts")

COHORTS = ("A", "B", "C")
# cohort B's identifiers are synthetic, so the packaged annotation cannot know them
MAPPING = {"B": os.path.join(DATA, "gene_mapping_reference.csv")}
EXPECTED_IDS = {"A": "symbol", "B": "ensembl", "C": "symbol"}
EXPECTED_SCALE = {"A": "log", "B": "fpkm_or_tpm", "C": "counts"}
MISSING_IN_B = {"CD3D", "CD3E", "CD8A", "CD8B", "GZMB"}


def load():
    expr = {n: pd.read_csv(os.path.join(DATA, f"Cohort_{n}_expression.csv"), index_col=0).T
            for n in COHORTS}
    clin = {n: pd.read_csv(os.path.join(DATA, f"Cohort_{n}_clinical.csv")) for n in COHORTS}
    mut = {}
    for n in COHORTS:
        path = os.path.join(DATA, f"Cohort_{n}_mutation.csv")
        if os.path.exists(path):
            mut[n] = pd.read_csv(path)
    return expr, clin, mut


def main():
    warnings.simplefilter("ignore")
    expr, clin, mut = load()
    print(f"cohorts: " + ", ".join(
        f"{n} {expr[n].shape[0]}g x {expr[n].shape[1]}s" for n in COHORTS) +
        f" | mutation for {sorted(mut)}\n")

    # ---- 1. every cohort is recognised for what it is ----
    print("detection")
    for n in COHORTS:
        ids = gene_id.detect_id_type(expr[n].index)["type"]
        scale = dp.detect_expression_type(expr[n])["type"]
        print(f"  {n}: identifiers={ids:8s} values={scale:12s}")
        assert ids == EXPECTED_IDS[n], (n, ids)
        assert scale == EXPECTED_SCALE[n], (n, scale)

    # ---- 2. coverage decides whether TPM is even meaningful ----
    cov = dp.assess_gene_coverage(expr["A"])
    print(f"\ncoverage vs the {cov['n_reference_genes']}-gene reference: {cov['coverage']:.2%} of it, "
          f"targeted panel = {cov['likely_targeted_panel']}")
    assert cov["likely_targeted_panel"], "an 80-gene panel must not be treated as a transcriptome"
    assert set(cov["incompatible_methods"]) >= {"ESTIMATE", "NetBio", "ISAFN"}

    # ---- 3. clinical encodings collapse onto one convention ----
    print("\nclinical harmonisation")
    for n in COHORTS:
        std, rep = dp.harmonize_clinical(clin[n], cohort=n, verbose=False)
        got = {k: v for k, v in rep["columns"].items() if v}
        print(f"  {n}: {rep['values'].get('response')} -> {int(std['resp'].sum())}/{len(std)} "
              f"responders, cancer={std['cancer_type'].dropna().unique().tolist()}")
        assert std["resp"].isin([0, 1]).all()
        assert set(std["sex"].dropna()) <= {"Male", "Female"}
        assert set(std["met1"].dropna()) <= {0.0, 1.0}
        assert got["sample"] in ("sample_id", "SampleID", "patient")
    # CR and PR are responders, SD and PD are not
    std_b, rep_b = dp.harmonize_clinical(clin["B"], verbose=False)
    assert rep_b["values"]["response"] == {"SD": 0.0, "PD": 0.0, "PR": 1.0, "CR": 1.0}
    # urothelial carcinoma is scored with ISAFN's bladder code
    std_c, _ = dp.harmonize_clinical(clin["C"], verbose=False)
    assert set(std_c["cancer_type"]) == {"bladder"}

    # ---- 4. the merge keeps provenance and does not invent values ----
    merged, mclin, mmut, rep = dp.merge_cohorts(expr, clin, mut, mapping=MAPPING, verbose=False)
    print(f"\nmerged: {merged.shape[0]} genes x {merged.shape[1]} samples, "
          f"{rep['n_missing_values']} missing values")
    assert list(mclin["source"].unique()) == list(COHORTS) or set(mclin["source"]) == set(COHORTS)
    assert set(rep["genes_missing_somewhere"]) == MISSING_IN_B
    # NaN appears only where cohort B did not measure the gene -- nowhere else
    b_samples = mclin.index[mclin["source"] == "B"]
    assert merged.loc[sorted(MISSING_IN_B), b_samples].isna().all().all()
    assert merged.drop(index=sorted(MISSING_IN_B)).notna().all().all()
    assert rep["cohorts_without_mutation_data"] == ["C"]
    print(f"  NaN confined to {sorted(MISSING_IN_B)} x cohort B; "
          f"cohorts without mutation data: {rep['cohorts_without_mutation_data']}")

    # ---- 5. every combination scores without raising ----
    print("\nscoring every combination")
    results = {}
    for r in (1, 2, 3):
        for combo in itertools.combinations(COHORTS, r):
            e = {n: expr[n] for n in combo}
            c = {n: clin[n] for n in combo}
            m = {n: mut[n] for n in combo if n in mut} or None
            ex, cl, mu, _ = dp.merge_cohorts(e, c, m, mapping=MAPPING, verbose=False)
            sc = TME_score.get_all_score(ex, cl, response_col="resp", source_col="source", df_mut=mu,
                                         gender_col="sex", drug_col="drug", met_col="met1",
                                         cancer_col="cancer_type")
            computed = [col for col in sc.columns if sc[col].notna().any()]
            name = "+".join(combo)
            results[name] = sc
            print(f"  {name:7s} {sc.shape[0]:3d} samples, {len(computed)}/{sc.shape[1]} scores "
                  f"computed, failed: {sorted(sc.attrs['failed_scores']) or 'none'}")
            assert sc.shape[0] == ex.shape[1]
            assert len(computed) > 0
            # a failure must come with a reason, never a silently wrong number
            for bad in sc.attrs["failed_scores"]:
                assert sc[bad].isna().all(), f"{bad} failed but still produced values"

    # ISAFN is the default score and must survive every combination
    for name, sc in results.items():
        assert "isafn_expr_prob" in sc.columns and sc["isafn_expr_prob"].notna().any(), name
    print("  ISAFN produced a score for all seven combinations")

    # ---- 6. per-cohort scoring vs scoring the merged matrix ----
    each = pd.concat([results[n]["isafn_expr_prob"] for n in COHORTS])
    both = results["A+B+C"]["isafn_expr_prob"]
    common = each.index.intersection(both.index)
    r = np.corrcoef(each.loc[common], both.loc[common])[0, 1]
    print(f"\nISAFN scored per cohort vs on the merged matrix: r = {r:.4f} over {len(common)} samples")

    # ---- 7. corrections, judged on signal kept AND cohort effect removed ----
    print("\nbatch correction")

    def score_fn(ex, cl):
        return TME_score.get_all_score(ex, cl, response_col="resp", source_col="source",
                                       df_mut=mmut, gender_col="sex", drug_col="drug",
                                       met_col="met1", cancer_col="cancer_type")

    summary, detail = optimal.compare_batch_correction(
        merged, mclin, score_fn, response_col="resp", source_col="source", verbose=True)
    unavailable = [m for m, d in detail.items() if "error" in d]
    if unavailable:
        print(f"  not available here: {unavailable}")
    assert len(summary) > 0
    print("\nall multi-cohort tests passed")


if __name__ == "__main__":
    main()
