"""Tests for scripts/hpa_dataset/preprocessing/check_dataset_splits.py (synthetic data)."""
import pandas as pd

import check_dataset_splits as C
import prepare_pilot_dataset as P

FRACS = (0.7, 0.1, 0.2)


def _frame(n_hpa=2000, n_cab=500):
    ids = [f"HPA{i:06d}" for i in range(1, n_hpa + 1)] + [f"CAB{i:06d}" for i in range(1, n_cab + 1)]
    rows = [{"image_id": f"{ab}_{t[:4]}", "antibody_id": ab, "ensembl_id": "G" + ab, "tissue": t}
            for ab in ids for t in P.PILOT_TISSUES[:5]]
    return pd.DataFrame(rows)


def _stained(df):
    num = df["antibody_id"].str[3:].astype(int)
    year = 2004 + (num * 15 // num.max()).clip(upper=14)       # older ids stained earlier
    return pd.Series((year.astype(str) + "-01-01").values, index=df["image_id"])


def test_v2_split_passes_all_checks():
    df = P.assign_sets(_frame(), *FRACS, seed=42)
    res = C.check_dataset("x", df, FRACS, 1.0, 1.5, stained=_stained(df))
    assert all(ok for _, ok, _ in res), [r for r in res if not r[1]]


def test_id_ordered_split_fails_strata_and_year_checks():
    df = _frame()
    order = df.drop_duplicates("antibody_id")["antibody_id"].sort_values(key=lambda s: s.str[3:].astype(int))
    cut = {a: ("test" if i < 0.2 * len(order) else "val" if i < 0.3 * len(order) else "train")
           for i, a in enumerate(order)}
    df["set"] = df["antibody_id"].map(cut)
    failed = {check for check, ok, _ in C.check_dataset("x", df, FRACS, 1.0, 1.5, stained=_stained(df)) if not ok}
    assert any(c.startswith("source x number") for c in failed) and any(c.startswith("staining-year") for c in failed)


def test_gene_agreement_and_main(tmp_path):
    a = P.assign_sets(_frame(400, 100), *FRACS, seed=42)
    b = a.copy()
    g = b["ensembl_id"].iloc[0]
    b.loc[b["ensembl_id"] == g, "set"] = "test" if a.loc[a["ensembl_id"] == g, "set"].iloc[0] != "test" else "train"
    assert [ok for _, ok, _ in C.check_gene_agreement({"a": a, "b": a})] == [True]
    assert [ok for _, ok, _ in C.check_gene_agreement({"a": a, "b": b})] == [False]
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    assert C.main(["--dataset", f"a={tmp_path / 'a.csv'}", "--strata-tol", "5"]) == 0
    assert C.main(["--dataset", f"a={tmp_path / 'a.csv'}", "--dataset", f"b={tmp_path / 'b.csv'}",
                   "--strata-tol", "5"]) == 1
