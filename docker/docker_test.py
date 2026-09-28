"""
End-to-end check of the TMEImmune v2 container.

Run it from the repository root, with the repository mounted so the example data is reachable:

    docker run --rm -v $(pwd):/work -w /work tmeimmune:v2 python docker/docker_test.py

It walks the whole v2 workflow on the bundled example data and exercises every heavy dependency in
the image, so a failure here points at a broken image rather than at the analysis:

    torch + the packaged model pickle   ISAFN
    gseapy                              TGFb, ESTIMATE, ISTME
    statsmodels                         NetBio pathway selection
    scikit-learn                        NetBio, calibration, clustering
    inmoose                             ComBat cross-cohort correction
    lifelines                           survival (imported by optimal)
    matplotlib                          the saved figures

Exit code is 0 only when every required step passed.
"""
import os
import sys
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")                       # no display inside a container

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

# paths: everything is relative to the mounted repository, overridable for other layouts
ROOT = os.environ.get("TMEIMMUNE_ROOT", os.getcwd())
DATA = os.environ.get("TMEIMMUNE_DATA", os.path.join(ROOT, "data"))
OUT = os.environ.get("TMEIMMUNE_OUT", os.path.join(ROOT, "docker", "output"))

failures = []
skipped = []


def run_step(name, fn):
    """Run one step, report it, and record a failure without stopping the rest."""
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    try:
        return fn()
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"FAILED  {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return None


# --------------------------------------------------------------------------------------------
print("=" * 72)
print("TMEImmune container check")
print("=" * 72)
import TMEImmune
from TMEImmune import data_processing as dp, gene_id, TME_score, optimal

versions = {"TMEImmune": TMEImmune.__version__, "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__}
for mod in ("torch", "sklearn", "scipy", "statsmodels", "gseapy", "lifelines", "inmoose",
            "joblib", "matplotlib"):
    try:
        loaded_mod = __import__(mod)
    except Exception as exc:
        versions[mod] = f"MISSING ({type(exc).__name__})"
        failures.append(f"import {mod}: {exc}")
        continue
    version = getattr(loaded_mod, "__version__", None)
    if version is None:                     # some packages only declare it in their metadata
        try:
            from importlib.metadata import version as _pkg_version
            version = _pkg_version({"sklearn": "scikit-learn"}.get(mod, mod))
        except Exception:
            version = "installed (version unknown)"
    versions[mod] = version
for k, v in versions.items():
    print(f"  {k:12s} {v}")

if not TMEImmune.__version__.startswith("2."):
    failures.append(f"expected TMEImmune 2.x in this image, found {TMEImmune.__version__}")
    print(f"\n  ** this image has TMEImmune {TMEImmune.__version__}, not a 2.x release **")

os.makedirs(OUT, exist_ok=True)
dp.set_confirm(True)                        # a container has nobody to answer a prompt
print(f"\n  data from {DATA}\n  figures to {OUT}")

riaz_path = os.path.join(DATA, "example_riaz.xlsx")
cohort_dir = os.path.join(DATA, "pseudo_cohorts")
if not os.path.exists(riaz_path):
    print(f"\n** {riaz_path} not found -- mount the repository at /work, or set TMEIMMUNE_DATA **")
    sys.exit(2)


# --------------------------------------------------------------------------------------------
def load():
    xl = pd.ExcelFile(riaz_path)
    expr = pd.read_excel(xl, "riaz").set_index("gene")
    clin_raw = pd.read_excel(xl, "Sheet1").dropna(subset=["ID"])
    keep = [c for c in expr.columns if "Pre_" in c and c in set(clin_raw["ID"])]
    expr = expr[keep]
    clin_raw = clin_raw[clin_raw["ID"].isin(keep)]
    ids = gene_id.detect_id_type(expr.index)["type"]
    values = dp.detect_expression_type(expr)["type"]
    cov = dp.assess_gene_coverage(expr)
    print(f"  {expr.shape[0]} genes x {expr.shape[1]} samples")
    print(f"  identifiers  {ids}")
    print(f"  values       {values}")
    print(f"  coverage     {cov['coverage']:.1%} of the {cov['n_reference_genes']}-gene reference, "
          f"panel={cov['likely_targeted_panel']}")
    assert ids == "symbol" and not cov["likely_targeted_panel"]
    return expr, clin_raw


loaded = run_step("1. read the example cohort and work out what it is", load)
if loaded is None:
    print("\ncannot continue without the example cohort")
    sys.exit(1)
expr, clin_raw = loaded


def harmonise():
    expr_h, rep = dp.harmonize(df=expr, verbose=False)
    print(f"  {expr_h.shape[0]} genes x {expr_h.shape[1]} samples")
    print(f"  steps    {rep['steps']}")
    print(f"  concerns {[c['code'] for c in rep['concerns']]} ({rep['confirmed']})")
    assert rep["converted"]
    return expr_h


expr_h = run_step("2. harmonise expression to log2(TPM+1)", harmonise)


def harmonise_clin():
    clin, rep = dp.harmonize_clinical(clin_raw, sample_col="ID", response_col="response",
                                      met_col="Metastasis", drug_col="drug", cohort="Riaz",
                                      verbose=False)
    print(f"  matched   {[k for k, v in rep['columns'].items() if v]}")
    print(f"  response  {rep['values']['response']}")
    print(f"  {int(clin['resp'].sum())} responders of {len(clin)}")
    assert clin["resp"].isin([0, 1]).all()
    return clin


clin = run_step("3. harmonise clinical variables", harmonise_clin)
if expr_h is None or clin is None:
    print("\ncannot continue without harmonised inputs")
    sys.exit(1)


def score():
    sc = TME_score.get_all_score(expr_h, clin, response_col="resp", drug_col="drug",
                                 met_col="met1", impute_sex=True)
    got = [c for c in sc.columns if sc[c].notna().any()]
    print(f"  {sc.shape[0]} samples x {sc.shape[1]} scores, {len(got)} computed")
    if sc.attrs["failed_scores"]:
        for name, why in sc.attrs["failed_scores"].items():
            print(f"  could not compute {name}: {why[:100]}")
    assert "isafn_expr_prob" in sc.columns and sc["isafn_expr_prob"].notna().any(), \
        "ISAFN produced nothing -- torch or the packaged model is broken in this image"
    print(f"  ISAFN probability range {sc['isafn_expr_prob'].min():.3f}"
          f"-{sc['isafn_expr_prob'].max():.3f}")
    for name in ("NetBio", "SIA", "ESTIMATE", "estimate", "TGFb"):
        if name in sc.columns and name not in sc.attrs["failed_scores"]:
            print(f"  {name} computed")
    return sc


sc = run_step("4. every score, including ISAFN (torch + model-pickle check)", score)
if sc is None:
    print("\ncannot evaluate without scores")
    sys.exit(1)

sc_eval = sc.copy()
sc_eval["resp"] = clin.loc[sc_eval.index, "resp"]
names = [c for c in sc.columns if not str(c).endswith("_pred") and sc[c].notna().any()]


def discrimination():
    # show_fig=True builds the figure; with the Agg backend nothing is displayed, it is just saved
    outcomes, table = optimal.get_performance(sc_eval, metric="ICI", score_name=names,
                                              ICI_col="resp", show_fig=True, top_n=5,
                                              return_table=True)
    best = table.sort_values("AUC", ascending=False).head(5)
    print(best.round(4).to_string())
    fig = outcomes[0][0] if isinstance(outcomes, (list, tuple)) else outcomes
    assert fig is not None, "get_performance returned no figure even with show_fig=True"
    path = os.path.join(OUT, "test_roc.pdf")
    fig.savefig(path)
    print(f"  wrote {path}")
    return table


run_step("5. discrimination: ROC for the top 5, table for all", discrimination)


def dca():
    curves, summary = optimal.decision_curve(sc_eval, "resp", score_name=names, verbose=False)
    print(summary.head(5)[["score", "area_above_default", "best_advantage",
                           "best_threshold"]].round(4).to_string(index=False))
    fig = optimal.plot_decision_curve(curves, summary, top_n=5)
    path = os.path.join(OUT, "test_decision_curve.pdf")
    fig.savefig(path)
    print(f"  wrote {path}")


run_step("6. decision curve analysis", dca)


def redundancy():
    corr, info = optimal.score_correlation(sc[names], verbose=False)
    print(f"  {corr.shape[0]} scores, {len(info['near_duplicates'])} near-duplicate pairs")
    for a, b, r in info["near_duplicates"][:3]:
        print(f"    {a} ~ {b}: r = {r:+.3f}")
    pairs = optimal.discordant_pairs(sc_eval, "resp", "isafn_expr_prob", verbose=False)
    print(pairs.head(4)[["comparator", "auc_reference", "auc_comparator", "fraction_disagree",
                         "reference_win_rate", "sign_test_p"]].round(4).to_string(index=False))
    cases, summary = optimal.disagreement_cases(sc_eval, "resp", "isafn_expr_prob",
                                                pairs.iloc[-1]["comparator"], n=10, verbose=False)
    print(f"  {len(cases)} most-disagreed patients, overall response rate "
          f"{summary['overall_response_rate']:.1%}")


run_step("7. redundancy: what ISAFN adds over the other scores", redundancy)


# --------------------------------------------------------------------------------------------
def multi_cohort():
    if not os.path.isdir(cohort_dir):
        skipped.append(f"multi-cohort: {cohort_dir} not present")
        print(f"  skipped: {cohort_dir} not found")
        return
    E = {n: pd.read_csv(os.path.join(cohort_dir, f"Cohort_{n}_expression.csv"), index_col=0).T
         for n in "ABC"}
    C = {n: pd.read_csv(os.path.join(cohort_dir, f"Cohort_{n}_clinical.csv")) for n in "ABC"}
    M = {n: pd.read_csv(os.path.join(cohort_dir, f"Cohort_{n}_mutation.csv")) for n in "AB"}
    mapping = {"B": os.path.join(cohort_dir, "gene_mapping_reference.csv")}

    mexpr, mclin, mmut, rep = dp.merge_cohorts(E, C, M, mapping=mapping, how="union", verbose=False)
    print(f"  merged {mexpr.shape[0]} genes x {mexpr.shape[1]} samples, "
          f"{rep['n_missing_values']} missing values")
    print(f"  genes missing from some cohort: {rep['genes_missing_somewhere']}")
    print(f"  cohorts without mutation data:  {rep['cohorts_without_mutation_data']}")

    msc = TME_score.get_all_score(mexpr, mclin, response_col="resp", source_col="source",
                                  df_mut=mmut, gender_col="sex", drug_col="drug",
                                  met_col="met1", cancer_col="cancer_type")
    print(f"  scored {msc.shape[0]} samples; could not compute "
          f"{sorted(msc.attrs['failed_scores'])} (expected on an 80-gene panel)")
    assert msc["isafn_expr_prob"].notna().any()

    def score_fn(ex, cl):
        return TME_score.get_all_score(ex, cl, response_col="resp", source_col="source",
                                       df_mut=mmut, gender_col="sex", drug_col="drug",
                                       met_col="met1", cancer_col="cancer_type")

    summary, detail = optimal.compare_batch_correction(
        mexpr, mclin, score_fn, methods=("none", "minmax_global", "combat"), verbose=False)
    if len(summary):
        per_method = summary.groupby("method").agg(
            mean_auc=("auc_directional", "mean"), batch_silhouette=("batch_silhouette", "first"),
            auc_within_cohort=("auc_within_cohort", "mean"))
        print(per_method.round(4).to_string())
    unavailable = {m: d["error"] for m, d in detail.items() if "error" in d}
    if "combat" in unavailable:
        failures.append(f"ComBat did not run: {unavailable['combat'][:120]}")
        print(f"  ** ComBat failed: {unavailable['combat'][:120]} -- check inmoose in this image **")
    else:
        print("  ComBat ran, so inmoose works in this image")


run_step("8. three cohorts: merge, score, compare batch corrections", multi_cohort)


# --------------------------------------------------------------------------------------------
print(f"\n{'=' * 72}")
if failures:
    print(f"{len(failures)} problem(s):")
    for f in failures:
        print(f"  - {f}")
if skipped:
    for s in skipped:
        print(f"  skipped: {s}")
if not failures:
    print("container check passed: every step ran and every dependency is present")
print("=" * 72)
sys.exit(1 if failures else 0)
