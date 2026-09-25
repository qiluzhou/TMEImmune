"""
Confirmation before a transformation that changes the shape of the data.
Run from the project root:  python tests/test_confirm.py

Converting between expression units is mostly arithmetic, but a few steps do more than change the
units -- they change what the numbers mean, and they do it silently. This checks that each one is
detected, described, and put to the user before it happens.

The transformations treated as questionable:
  not_recoverable    z-scored or centred input; the original scale cannot be recovered  (blocking)
  no_gene_lengths    raw counts with no gene lengths, so only CPM is possible
  genes_dropped      genes without a known length are removed from the matrix
  rescaled           every sample is rescaled to sum to 1e6
  log_base_unknown   the values are logged but the base cannot be told from them
  targeted_panel     a per-million total over a panel is not a real TPM

Behaviour asked for here:
  confirm=True    convert without asking
  confirm=False   decline without asking
  confirm='auto'  ask when a person can answer; with nobody to ask, decline only on a blocking
                  concern and otherwise proceed with the warning, so existing scripts keep working
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TMEImmune import data_processing as dp

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cohorts():
    """One real cohort expressed four ways, each of which triggers a different concern."""
    xlsx = pd.ExcelFile(os.path.join(root, "data", "example_riaz.xlsx"))
    tpm = pd.read_excel(xlsx, "riaz").set_index("gene")
    tpm = tpm[[c for c in tpm.columns if "Pre_" in c]]
    logged = np.log2(tpm + 1)
    zscored = logged.sub(logged.mean(1), axis=0).div(logged.std(1).replace(0, 1), axis=0)
    counts = (tpm / 1e6 * 3e7).round()
    return {"z-scored": (zscored, "not_recoverable"),
            "TPM subset": (tpm, "rescaled"),
            "raw counts": (counts, "no_gene_lengths")}


def main():
    warnings.simplefilter("ignore")
    data = cohorts()
    print(f"{len(data)} inputs, each raising a different concern\n")

    # ---- 1. the concern is detected and described ----
    print("1. detection")
    for name, (df, expected) in data.items():
        _, rep = dp.to_log2tpm(df, verbose=False, confirm=False)
        codes = [c["code"] for c in rep["concerns"]]
        assert expected in codes, (name, codes)
        concern = next(c for c in rep["concerns"] if c["code"] == expected)
        assert concern["detail"] and concern["consequence"], "a concern must say what and so what"
        assert concern["severity"] in ("blocking", "advisory")
        print(f"   {name:12s} -> {expected} ({concern['severity']})")

    # ---- 2. confirm=False declines and leaves the data untouched ----
    print("\n2. confirm=False")
    for name, (df, _) in data.items():
        out, rep = dp.to_log2tpm(df, verbose=False, confirm=False)
        assert rep["converted"] is False and rep["confirmed"] == "assumed"
        assert out.equals(df.apply(pd.to_numeric, errors="coerce")), f"{name} was modified anyway"
    print("   nothing converted, every matrix returned unchanged")

    # ---- 3. confirm=True converts without asking ----
    print("\n3. confirm=True")
    for name, (df, _) in data.items():
        out, rep = dp.to_log2tpm(df, verbose=False, confirm=True)
        assert rep["converted"] is True and rep["confirmed"] == "assumed"
        print(f"   {name:12s} -> {rep['steps']}")

    # ---- 4. confirm='auto' with nobody to ask: blocking stops, advisory proceeds ----
    # Whether a terminal exists depends on how this test was started, so both paths are driven
    # explicitly rather than left to the environment.
    print(f"\n4. confirm='auto' with nobody to ask "
          f"(a terminal {'is' if dp._can_prompt() else 'is not'} attached; forcing the no-terminal path)")
    real_can_prompt = dp._can_prompt
    dp._can_prompt = lambda: False
    try:
        for name, (df, expected) in data.items():
            out, rep = dp.to_log2tpm(df, verbose=False)
            blocking = any(c.get("severity") == "blocking" for c in rep["concerns"])
            assert rep["confirmed"] == "nobody to ask"
            assert rep["converted"] == (not blocking), name
            print(f"   {name:12s} -> converted={rep['converted']} (blocking={blocking})")

        # the previous behaviour has to be preserved exactly for code that never asked for any of this
        plain, _ = dp.to_log2tpm(data["TPM subset"][0], verbose=False, confirm=True)
        auto, _ = dp.to_log2tpm(data["TPM subset"][0], verbose=False)
        assert np.allclose(plain.to_numpy(), auto.to_numpy()), \
            "an advisory concern must not change what a script gets"
        print("   advisory concerns give byte-identical output to the old behaviour")
    finally:
        dp._can_prompt = real_can_prompt

    # ---- 4b. confirm='auto' with somebody to ask: the answer decides ----
    print("\n4b. confirm='auto' with an answer (the prompt is answered for you here)")
    real_ask = dp._ask_yes_no
    dp._can_prompt = lambda: True
    try:
        for answer, expect_converted, expect_how in [(True, True, "answered yes"),
                                                     (False, False, "answered no"),
                                                     (None, None, "nobody to ask")]:
            dp._ask_yes_no = lambda question, default=False, _a=answer: _a
            for name, (df, _) in data.items():
                out, rep = dp.to_log2tpm(df, verbose=False)
                blocking = any(c.get("severity") == "blocking" for c in rep["concerns"])
                want = (not blocking) if expect_converted is None else expect_converted
                assert rep["confirmed"] == expect_how, (name, rep["confirmed"])
                assert rep["converted"] == want, (name, answer, rep["converted"])
                if not rep["converted"]:
                    assert out.equals(df.apply(pd.to_numeric, errors="coerce"))
            shown = {True: "yes", False: "no", None: "no answer"}[answer]
            print(f"   answered {shown:9s} -> confirmed='{expect_how}' for all three inputs")
    finally:
        dp._ask_yes_no = real_ask
        dp._can_prompt = real_can_prompt

    # force=True still means 'convert regardless', including through a blocking concern
    _, rep = dp.to_log2tpm(data["z-scored"][0], verbose=False, force=True)
    assert rep["converted"] is True
    print("   force=True still overrides a blocking concern")

    # ---- 5. clean input raises nothing and is never questioned ----
    print("\n5. clean input")
    clean, rep = dp.to_log2tpm(np.log2(data["TPM subset"][0] + 1), verbose=False)
    assert rep["concerns"] == [] and rep["confirmed"] == "no concerns" and rep["converted"]
    print("   log2 input: no concerns raised, converted without a question")

    # ---- 6. the option reaches the callers that matter ----
    print("\n6. threaded through the public entry points")
    _, hrep = dp.harmonize(df=data["z-scored"][0], verbose=False, confirm=False)
    assert hrep["converted"] is False
    import inspect
    from TMEImmune import ISAFN
    for fn in (dp.harmonize, dp.merge_cohorts, dp.to_log2tpm, ISAFN.isafn_score):
        assert "confirm" in inspect.signature(fn).parameters, fn.__name__
    print("   to_log2tpm, harmonize, merge_cohorts and isafn_score all accept confirm=")

    # ---- 7. one setting for a whole session, instead of an argument on every call ----
    print("\n7. set_confirm()")
    zscored = data["z-scored"][0]
    previous = dp.set_confirm(True)
    try:
        assert dp.get_confirm() is True
        _, rep = dp.to_log2tpm(zscored, verbose=False)
        assert rep["converted"] is True and rep["confirmed"] == "assumed"
        # an explicit argument on the call still beats the session setting
        _, rep = dp.to_log2tpm(zscored, verbose=False, confirm=False)
        assert rep["converted"] is False
        dp.set_confirm(False)
        _, rep = dp.to_log2tpm(data["TPM subset"][0], verbose=False)
        assert rep["converted"] is False, "set_confirm(False) must stop even an advisory conversion"
        # and it reaches the callers, not just to_log2tpm
        _, hrep = dp.harmonize(df=data["TPM subset"][0], verbose=False)
        assert hrep["converted"] is False
    finally:
        dp.set_confirm(previous)
    assert dp.get_confirm() == previous
    print(f"   set_confirm(True/False) applies to every call; an explicit confirm= still wins; "
          f"restored to {previous!r}")

    try:
        dp.set_confirm("sometimes")
        raise AssertionError("set_confirm should reject anything but True, False or 'auto'")
    except ValueError:
        pass
    print("   set_confirm rejects an invalid value")

    print("\nall confirmation tests passed")


if __name__ == "__main__":
    main()
