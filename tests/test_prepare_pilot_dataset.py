"""Tests for scripts/hpa_dataset/preprocessing/prepare_pilot_dataset.py (synthetic data)."""
import pandas as pd
import pytest

import prepare_pilot_dataset as P


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _img(image_id, antibody, gene, tissue, patient="4016", itype="tif", organ="Organ"):
    return {
        "antibody_id": antibody, "ensembl_id": gene, "gene_name": gene.lower(),
        "tissue": tissue, "organ": organ, "patient_id": patient, "image_type": itype,
        "image_url": f"https://images.proteinatlas.org/1/{image_id}.{itype}",
        "image_id": image_id, "local_path": "20250612-xml/1",
        "download_status": "pending", "download_date": "", "file_size": "", "url_used": "",
    }


@pytest.fixture
def antibodies():
    rows = [
        ("HPA1", "G1", "approved"),
        ("HPA2", "G1", "enhanced"),      # best for G1 but has NO images
        ("HPA3", "G2", "supported"),
        ("HPA4", "G2", "supported"),     # tie with HPA3
        ("HPA5", "G3", "uncertain"),
        ("HPA6", "G4", "approved"), ("HPA6", "G5", "approved"),  # multi-target
        ("HPA7", "G5", "approved"),
    ]
    return pd.DataFrame([{"antibody_id": a, "ensembl_id": g, "gene_name": g, "Reliability score": r}
                         for a, g, r in rows])


@pytest.fixture
def images():
    tissues = P.PILOT_TISSUES[:6]
    rows = []
    n = 0
    for ab, gene in (("HPA1", "G1"), ("HPA3", "G2"), ("HPA4", "G2"), ("HPA5", "G3"),
                     ("HPA6", "G4"), ("HPA6", "G5"), ("HPA7", "G5")):
        for t in tissues:
            for k in range(3):
                n += 1
                rows.append(_img(f"img{n:03d}_{ab}", ab, gene, t, patient=str(1000 + k)))
    # HPA6 rows: the same image ids appear once per target gene (as in the real crawl)
    df = pd.DataFrame(rows)
    df.loc[df["antibody_id"] == "HPA6", "image_id"] = (
        df.loc[df["antibody_id"] == "HPA6"].groupby("ensembl_id").cumcount().map(lambda i: f"img_hpa6_{i:02d}")
    )
    # a few JPG-only rows and a secondary-tissue row
    df.loc[df.index[:2], "image_type"] = "jpg"
    df.loc[df.index[2], "tissue"] = "Skin 2"
    return df


@pytest.fixture
def idr_table(images):
    # half of the images are mirrored on IDR, including the first JPG-only one
    ids = sorted(images["image_id"].unique())[::2]
    ids.append(images.loc[images.index[0], "image_id"])
    rows = [{"Image Name": f"{i}.tif", "ftp_path": f"20180825-ftp/1/{i}.tif"} for i in set(ids)]
    rows.append({"Image Name": "orphan.tif", "ftp_path": ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stage A
# --------------------------------------------------------------------------- #

def test_read_csv_keeps_ids_as_strings(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("image_id,patient_id\n1_A,4016\n2_B,\n")
    df = P.read_csv(p)
    assert df["patient_id"].tolist() == ["4016", ""]
    assert df["patient_id"].dtype == object


def test_attach_idr_builds_urls(images, idr_table):
    df = P.attach_idr(images, idr_table)
    assert "hpa_url" in df.columns and "image_url" not in df.columns
    on = df[df["idr_available"]]
    off = df[~df["idr_available"]]
    assert len(on) and len(off)
    assert (on["idr_url"] == P.IDR_HTTPS_BASE + on["idr_ftp_path"]).all()
    assert (off["idr_url"] == "").all() and (off["idr_ftp_path"] == "").all()
    # empty ftp_path rows in the IDR table never match
    assert "orphan" not in set(df.loc[df["idr_available"], "image_id"])


def test_source_rule_and_tif_urls(images, idr_table):
    df = P.attach_idr(images, idr_table)
    idr_only = P.filter_source(df, "idr")
    assert idr_only["idr_available"].all()
    assert len(P.filter_source(df, "all")) == len(df)
    assert df["hpa_url"].str.endswith(".tif").all()      # JPG-only crawl rows still get the .tif link
    assert (df["image_type"] == "jpg").any()               # ...while image_type keeps what the XML listed


def test_antibody_filters_order(images, antibodies):
    df = P.attach_idr(images, None)
    steps = P.StepLog("t")
    out = P.filter_antibodies(df, antibodies, seed=1, steps=steps)
    kept = set(out["antibody_id"])
    assert "HPA5" not in kept                       # uncertain
    assert "HPA6" not in kept                       # multi-target
    assert "HPA1" in kept                           # G1: HPA2 is better but has no images
    assert "HPA7" in kept                           # G5 keeps its specific antibody
    assert len({"HPA3", "HPA4"} & kept) == 1        # one antibody for G2
    assert out.groupby("ensembl_id")["antibody_id"].nunique().max() == 1


def test_normalize_patient_id():
    df = pd.DataFrame({"patient_id": ["1957.0", "4016", "", "12.5", "x.0"]})
    out, n = P.normalize_patient_id(df)
    assert out["patient_id"].tolist() == ["1957", "4016", "", "12.5", "x.0"] and n == 1


def test_best_antibody_tie_break_is_seeded(antibodies):
    cands = antibodies[antibodies["ensembl_id"] == "G2"]
    a = P.select_best_antibody_per_gene(cands, seed=3)["antibody_id"].iloc[0]
    b = P.select_best_antibody_per_gene(cands, seed=3)["antibody_id"].iloc[0]
    assert a == b


def test_unknown_reliability_score_raises(antibodies):
    bad = antibodies.copy()
    bad.loc[0, "Reliability score"] = "weird"
    with pytest.raises(ValueError):
        P.select_best_antibody_per_gene(bad, seed=0)


def test_duplicates_fail_prefer_and_drop(tmp_path):
    df = pd.DataFrame([_img("a", "HPA1", "G1", "Lung"), _img("a", "HPA1", "G1", "Colon"),
                       _img("b", "HPA1", "G1", "Adipose tissue"), _img("b", "HPA1", "G1", "Soft tissue 1"),
                       _img("c", "HPA1", "G1", "Lung")])
    report = tmp_path / "dups.csv"
    with pytest.raises(SystemExit):
        P.resolve_duplicates(df, "fail", str(report))
    assert report.exists() and len(pd.read_csv(report)) == 4
    pref = P.resolve_duplicates(df, "prefer", str(report))
    assert sorted(pref["image_id"]) == ["b", "c"]
    assert pref.loc[pref["image_id"] == "b", "tissue"].item() == "Soft tissue 1"
    assert P.resolve_duplicates(df, "drop", str(report))["image_id"].tolist() == ["c"]
    assert P.resolve_duplicates(df[df["image_id"] == "c"], "fail", str(report)).shape[0] == 1


def _split_frame(n_antibodies=40, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n_antibodies):
        ab = f"HPA{i:03d}"
        for t in rng.sample(P.PILOT_TISSUES, rng.randint(3, 8)):
            for k in range(2):
                rows.append(_img(f"{ab}_{t[:4]}_{k}", ab, f"G{i}", t))
    return P.attach_idr(pd.DataFrame(rows), None)


def test_split_is_seeded_and_antibody_disjoint():
    df = _split_frame()
    a = P.assign_sets(df, 0.7, 0.1, 0.2, seed=42)
    b = P.assign_sets(df, 0.7, 0.1, 0.2, seed=42)
    assert a["set"].tolist() == b["set"].tolist()
    assert set(a["set"]) == {"train", "val", "test"}
    assert (a.groupby("antibody_id")["set"].nunique() == 1).all()
    frac = a["set"].value_counts(normalize=True)
    assert 0.55 < frac["train"] < 0.85 and 0.10 < frac["test"] < 0.35
    c = P.assign_sets(df, 0.7, 0.1, 0.2, seed=7)
    assert c["set"].tolist() != a["set"].tolist()


def test_split_from_inherits_and_fills(images, tmp_path):
    df = P.attach_idr(images, None)
    prev = df.copy()
    prev["set"] = "test"
    prev = prev[prev["antibody_id"] == "HPA1"].iloc[:4][["image_id", "set"]]   # partial antibody
    p = tmp_path / "prev.csv"
    prev.to_csv(p, index=False)
    out = P.assign_sets(df, 0.7, 0.1, 0.2, seed=42, split_from=str(p))
    assert (out.loc[out["antibody_id"] == "HPA1", "set"] == "test").all()   # filled from antibody
    assert (out.groupby("antibody_id")["set"].nunique() == 1).all()


def test_full_stage_end_to_end(images, antibodies, idr_table, tmp_path):
    steps = P.StepLog("full")
    out = P.build_full_dataset(images, antibodies, idr_table, "idr", seed=42,
                               duplicate_report=str(tmp_path / "d.csv"), steps=steps)
    final = P.finalize_columns(out)
    assert list(final.columns) == P.OUTPUT_COLUMNS
    assert final["idr_available"].all()
    assert "download_status" not in final.columns
    assert "Skin 2" not in set(final["tissue"])
    assert steps.frame()["step"].tolist()[0] == "crawled images"


# --------------------------------------------------------------------------- #
# Stage B
# --------------------------------------------------------------------------- #

def _pilot_frame():
    rows = []
    for ab in ("A", "B", "C", "D"):
        for t in P.PILOT_TISSUES[:5]:
            for k in range(3):
                rows.append(_img(f"{ab}_{t[:3]}_{k}", ab, f"G{ab}", t))
    rows.append(_img("E_x_0", "E", "GE", "Kidney"))   # not a pilot tissue
    return pd.DataFrame(rows)


def test_thin_triplets_keeps_two_of_three():
    df = _pilot_frame()
    out = P.thin_triplets(df, seed=0)
    sizes = out.groupby(P.GROUP_KEY).size()
    assert set(sizes[sizes.index.get_level_values("tissue").isin(P.PILOT_TISSUES)]) == {2}
    assert P.thin_triplets(df, seed=0).index.tolist() == out.index.tolist()   # deterministic
    assert list(out.columns) == list(df.columns)


def test_thin_pairs_drops_about_40_percent():
    df = P.thin_triplets(_pilot_frame(), seed=0)
    pairs_before = (df.groupby(P.GROUP_KEY).size() == 2).sum()
    out = P.thin_pairs(df, seed=0, drop_prob=P.PAIR_DROP_PROB)
    sizes = out.groupby(P.GROUP_KEY).size()
    assert set(sizes) <= {1, 2}
    dropped = (sizes == 1).sum() - 1                    # minus the single Kidney row
    assert 0 < dropped < pairs_before
    assert P.thin_pairs(df, seed=0).index.tolist() == out.index.tolist()
    assert len(P.thin_pairs(df, seed=0, drop_prob=0.0)) == len(df)
    assert (P.thin_pairs(df, seed=0, drop_prob=1.0).groupby(P.GROUP_KEY).size() == 1).all()


def test_pilot_tissue_list_is_frozen_from_bins():
    assert P.sample_pilot_tissues() == P.PILOT_TISSUES
    assert len(P.PILOT_TISSUES) == 20
    assert set(TISSUE for TISSUE in P.TISSUE_DIFFICULTY["difficult"]) <= set(P.PILOT_TISSUES)


def test_pilot_stage_end_to_end():
    df = _pilot_frame()
    df["set"] = "train"
    steps = P.StepLog("pilot")
    out = P.build_pilot_subset(df, seed=42, exclude_ids=["A_Adr_0"], steps=steps)
    assert set(out["tissue"]) <= set(P.PILOT_TISSUES)
    assert "A_Adr_0" not in set(out["image_id"])
    assert set(out["set"]) <= {"train", "val", "test"}
    assert (out.groupby("antibody_id")["set"].nunique() == 1).all()


# --------------------------------------------------------------------------- #
# Split v2: era/source balance, inheritance by gene, guards
# --------------------------------------------------------------------------- #

def _uniform_frame(n_hpa=300, n_cab=0, tissues=None, prefix=""):
    """Antibodies that all share the same tissues (as ~90 % of real ones do); higher HPA number = newer."""
    tissues = tissues or P.PILOT_TISSUES[:5]
    ids = [f"HPA{i:06d}" for i in range(1, n_hpa + 1)] + [f"CAB{i:06d}" for i in range(1, n_cab + 1)]
    rows = [{"image_id": f"{prefix}{ab}_{t[:4]}", "antibody_id": ab, "ensembl_id": "G" + ab, "tissue": t}
            for ab in ids for t in tissues]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_split_is_not_ordered_by_antibody_id(seed):
    # Regression for v1: skmultilearn on the ID-sorted antibody list put none of the newest half into test.
    s = pd.Series(P.split_antibodies(_uniform_frame(300), 0.7, 0.1, 0.2, seed))
    newest = s.index.str[3:].astype(int) > 150
    assert 0.4 <= (s[newest] == "test").sum() / (s == "test").sum() <= 0.6


def test_split_balances_antibody_source_and_fractions():
    s = pd.Series(P.split_antibodies(_uniform_frame(400, 100), 0.7, 0.1, 0.2, 42))
    cab = s.index.str.startswith("CAB")
    assert all(abs(cab[s == k].mean() - 0.2) <= 0.02 for k in P.SET_NAMES)
    assert s.value_counts().to_dict() == {"train": 350, "test": 100, "val": 50}


def test_split_depends_on_antibodies_and_seed_only():
    df = _uniform_frame(120, 30)
    a = P.split_antibodies(df, 0.7, 0.1, 0.2, 42)
    assert a == P.split_antibodies(df.sample(frac=1, random_state=3), 0.7, 0.1, 0.2, 42)
    assert a != P.split_antibodies(df, 0.7, 0.1, 0.2, 7)


def test_antibody_strata_fall_back_for_small_and_unparseable_ids():
    ids = [f"HPA{i:06d}" for i in range(40)] + ["CAB000001", "CAB000002", "CAB000003", "XYZ", "A"]
    assert pd.Series(P.antibody_strata(ids)).value_counts().min() >= P.MIN_STRATUM
    s = P.split_antibodies(pd.DataFrame({"antibody_id": ids, "tissue": "Lung"}), 0.7, 0.1, 0.2, 42)
    assert sorted(s) == sorted(ids) and set(s.values()) == set(P.SET_NAMES)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 7])
def test_split_small_inputs(n):
    ids = [chr(65 + i) for i in range(n)]
    s = P.split_antibodies(pd.DataFrame({"antibody_id": ids, "tissue": "Lung"}), 0.7, 0.1, 0.2, 42)
    assert sorted(s) == ids
    assert set(s.values()) <= ({"train"} if n < 3 else set(P.SET_NAMES))
    if n >= 3:
        assert "train" in s.values() and "test" in s.values()


def test_split_from_inherits_by_gene_and_splits_new_genes(tmp_path):
    pilot = P.assign_sets(_uniform_frame(100), 0.7, 0.1, 0.2, seed=42)
    path = tmp_path / "pilot.csv"
    pilot.to_csv(path, index=False)
    # full: more tissues, 50 new genes, and gene GHPA000001 measured with another antibody and other images
    full = _uniform_frame(150, tissues=P.PILOT_TISSUES[:8], prefix="f")
    full.loc[full["antibody_id"] == "HPA000001", "antibody_id"] = "CAB999999"
    out = P.assign_sets(full, 0.7, 0.1, 0.2, seed=42, split_from=str(path))
    g_pilot = pilot.drop_duplicates("ensembl_id").set_index("ensembl_id")["set"]
    g_full = out.drop_duplicates("ensembl_id").set_index("ensembl_id")["set"]
    assert (g_full[g_pilot.index] == g_pilot).all()
    assert set(out.loc[~out["ensembl_id"].isin(g_pilot.index), "set"]) == set(P.SET_NAMES)
    assert (out.groupby("antibody_id")["set"].nunique() == 1).all()
    assert (out["split_version"] == P.SPLIT_VERSION).all()


def test_split_from_conflicts_abort(tmp_path):
    df = _uniform_frame(30)
    prev = P.assign_sets(df, 0.7, 0.1, 0.2, seed=1)

    bad_ab = prev.copy()
    rows = bad_ab.index[bad_ab["antibody_id"] == "HPA000001"]
    bad_ab.loc[rows[:2], "set"], bad_ab.loc[rows[2:], "set"] = "train", "test"
    bad_ab.to_csv(tmp_path / "a.csv", index=False)
    with pytest.raises(RuntimeError, match="antibodies have images in several sets"):
        P.assign_sets(df, 0.7, 0.1, 0.2, seed=1, split_from=str(tmp_path / "a.csv"))

    per_ab = prev.drop_duplicates("antibody_id").set_index("antibody_id")["set"]
    a1, a2 = per_ab[per_ab == "train"].index[0], per_ab[per_ab == "test"].index[0]
    bad_gene = prev.copy()
    bad_gene.loc[bad_gene["antibody_id"] == a2, "ensembl_id"] = "G" + a1
    bad_gene.to_csv(tmp_path / "g.csv", index=False)
    with pytest.raises(RuntimeError, match="genes span several sets"):
        P.assign_sets(df, 0.7, 0.1, 0.2, seed=1, split_from=str(tmp_path / "g.csv"))

    empty = prev.copy()
    empty.loc[empty.index[0], "ensembl_id"] = ""
    empty.to_csv(tmp_path / "e.csv", index=False)
    with pytest.raises(ValueError, match="empty ensembl_id"):
        P.assign_sets(df, 0.7, 0.1, 0.2, seed=1, split_from=str(tmp_path / "e.csv"))


def test_split_from_version_guard(tmp_path):
    v2 = P.assign_sets(_uniform_frame(10), 0.7, 0.1, 0.2, seed=42)
    v2.to_csv(tmp_path / "v2.csv", index=False)
    P.check_split_from_version(str(tmp_path / "v2.csv"))
    v2.drop(columns="split_version").to_csv(tmp_path / "v1.csv", index=False)
    with pytest.raises(ValueError, match="split_version"):
        P.check_split_from_version(str(tmp_path / "v1.csv"))
    P.check_split_from_version(str(tmp_path / "v1.csv"), allow_legacy=True)


def test_main_split_version_set_v1_and_legacy_guard(images, antibodies, idr_table, tmp_path):
    for name, frame in (("images", images), ("antibodies", antibodies), ("idr", idr_table)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    base = ["--images-csv", str(tmp_path / "images.csv"), "--antibodies-csv", str(tmp_path / "antibodies.csv"),
            "--idr-table", str(tmp_path / "idr.csv"), "--source", "idr", "--stage", "full"]
    first = tmp_path / "first.csv"
    assert P.main(base + ["--full-output", str(first)]) == 0
    out = P.read_csv(first)
    assert list(out.columns) == P.OUTPUT_COLUMNS and set(out["split_version"]) == {P.SPLIT_VERSION}

    legacy = out.drop(columns="split_version").assign(set="test")
    legacy.to_csv(tmp_path / "legacy.csv", index=False)
    with pytest.raises(ValueError, match="split_version"):
        P.main(base + ["--full-output", str(tmp_path / "x.csv"), "--split-from", str(tmp_path / "legacy.csv")])

    second = tmp_path / "second.csv"
    assert P.main(base + ["--full-output", str(second), "--split-from", str(first),
                          "--set-v1-from", str(tmp_path / "legacy.csv")]) == 0
    again = P.read_csv(second)
    assert again["set"].tolist() == out["set"].tolist()
    assert list(again.columns) == P.OUTPUT_COLUMNS + ["set_v1"] and set(again["set_v1"]) == {"test"}
