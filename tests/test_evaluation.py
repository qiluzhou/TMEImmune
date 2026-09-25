"""
Decision curve analysis and redundancy analysis.   python tests/test_evaluation.py

AUC says how well a score orders patients. These two say whether acting on it would help, and whether
it tells you anything the scores you already have do not.

  1. net benefit satisfies its defining identities (perfect predictor, treat-all, treat-none)
  2. calibration does not depend on a score's arbitrary units
  3. decision curves and the summary ranking
  4. rank correlation and clustering across scores
  5. discordant-pair comparison, with a sign test
  6. the individual patients two scores disagree about

Nothing here fits a model to produce a score; the only fit is the single-variable logistic that puts
each score on a probability scale, which is what net benefit is defined against.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import ISAFN, optimal, TME_score as ts
from TMEImmune.optimal import net_benefit, _calibrate

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = 'isafn_expr_prob'


def build_scores():
    """Riaz pre-treatment samples: ISAFN plus a few signature scores to compare it against."""
    xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
    tpm = pd.read_excel(xlsx, "riaz").set_index("gene")
    clin = pd.read_excel(xlsx, "Sheet1").dropna(subset=["ID"]).set_index("ID")
    clin["resp"] = (clin["response"] == "R").astype(int)
    keep = [c for c in tpm.columns if "Pre_" in c and c in clin.index]
    tpm, clin = tpm[keep], clin.loc[keep]

    isafn = ISAFN.isafn_score(tpm.T, clin, drug_col="drug", met_col="Metastasis", impute_sex=True)
    logged = np.log2(tpm + 1)
    sc = pd.DataFrame(index=tpm.columns)
    for name, sig in [("CYT1", ts.CYT1), ("CYT2", ts.CYT2), ("TLS", ts.TLS), ("TIS", ts.TIS)]:
        try:
            sc[name] = ts.get_geomean_score(logged, sig, name)
        except Exception as exc:
            print(f"  ({name} unavailable: {str(exc)[:60]})")
    sc[REFERENCE] = isafn["isafn_expr_prob"]
    sc["resp"] = clin.loc[sc.index, "resp"]
    return sc.dropna(axis=1, how="all")


def check_net_benefit():
    """The three cases where net benefit has a known closed form."""
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.3).astype(int)
    prev = y.mean()
    th = np.array([0.1, 0.2, 0.3, 0.5, 0.8])

    # treating exactly the responders and no one else: every threshold gives the response rate
    assert np.allclose(net_benefit(y.astype(float), y, th), prev)
    # treating everyone: the 'Treat all' line
    assert np.allclose(net_benefit(np.ones(n), y, th), prev - (1 - prev) * th / (1 - th))
    # treating no one: zero everywhere
    assert np.allclose(net_benefit(np.zeros(n), y, th), 0)
    # and at a threshold equal to the response rate, treating everyone is worth exactly nothing
    assert abs(net_benefit(np.ones(n), y, [prev])[0]) < 1e-12
    print("net benefit: perfect predictor, treat-all, treat-none and the break-even point all hold")


def check_calibration_scale(sc):
    """A score's units must not change its decision curve."""
    y = sc["resp"]
    col = REFERENCE
    a = _calibrate(sc[col], y)
    b = _calibrate(sc[col] * 1000 + 7, y)
    assert np.allclose(a, b, atol=1e-6), "calibration is not invariant to the score's scale"
    # a narrow-range score must still spread out across risks rather than collapsing to a constant
    assert a.max() - a.min() > 0.05, f"calibrated risks collapsed to {a.min():.3f}-{a.max():.3f}"
    print(f"calibration is scale-invariant; {col} spans risk {a.min():.3f}-{a.max():.3f}")


def main():
    warnings.simplefilter("ignore")
    check_net_benefit()

    sc = build_scores()
    y = sc["resp"]
    print(f"\n{sc.shape[1] - 1} scores over {len(sc)} samples, "
          f"{int(y.sum())} responders ({y.mean():.1%})\n")
    check_calibration_scale(sc)

    # ---- decision curve ----
    print("\n1. decision curve analysis")
    curves, summary = optimal.decision_curve(sc, "resp")
    assert set(["Treat all", "Treat none"]).issubset(curves.columns)
    assert len(summary) == sc.shape[1] - 1
    # every curve must sit at or below the perfect-prediction ceiling, the response rate
    assert curves.drop(columns=["Treat none"]).max().max() <= y.mean() + 1e-9
    fig = optimal.plot_decision_curve(curves, summary, top_n=5,
                                      path=os.path.join(root, "tests", "decision_curve.png"),
                                      name="Riaz (pre-treatment)")

    print("\n   with out-of-fold calibration (cv=5), which removes most of the optimism:")
    _, cv_summary = optimal.decision_curve(sc, "resp", cv=5, verbose=False)
    print(cv_summary[["score", "area_above_default", "best_advantage"]].round(4).to_string(index=False))

    # ---- redundancy ----
    print("\n2. how much the scores duplicate each other")
    corr, info = optimal.score_correlation(sc.drop(columns=["resp"]))
    assert corr.shape[0] == sc.shape[1] - 1
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    print(corr.round(3).to_string())

    print("\n3. discordant pairs")
    dp = optimal.discordant_pairs(sc, "resp", REFERENCE)
    assert len(dp) == sc.shape[1] - 2
    # the pair counting is its own AUC implementation, so check it against sklearn's
    from sklearn.metrics import roc_auc_score
    for _, row in dp.iterrows():
        assert abs(row["auc_reference"] - roc_auc_score(y, sc[REFERENCE])) < 1e-9
        assert abs(row["auc_comparator"] - roc_auc_score(y, sc[row["comparator"]])) < 1e-9
        assert abs(row["auc_difference"] -
                   (row["auc_reference"] - row["auc_comparator"])) < 1e-12
        assert row["n_disagree"] == row["reference_wins"] + row["comparator_wins"]
    print("   pairwise AUCs match sklearn exactly; win/loss counts are consistent")

    print("\n4. the patients two scores disagree about")
    closest = dp.iloc[-1]["comparator"]
    cases, summ = optimal.disagreement_cases(sc, "resp", REFERENCE, closest, n=10)
    assert len(cases) > 0 and "response" in cases.columns

    print("\nall evaluation tests passed")


if __name__ == "__main__":
    main()
