#!/usr/bin/env python3
"""
HPA tissue dataset: build the full dataset and the pilot subset with splits.

The script partitions the *whole crawled* HPA image table (``images.csv`` from
``hpa_xml_parser.py``). It never looks at what has been downloaded locally, so
the resulting CSVs can be handed to collaborators who download the images
themselves from the URL columns.

Stage A - full dataset
    1. Attach IDR mirror info (``idr_ftp_path``, ``idr_available``, ``idr_url``)
       from the IDR normal-tissue table (see ``build_idr_ftp_paths.py``).
    2. ``--source idr``  keep only images mirrored on IDR
       ``--source all``  keep every crawled image (HPA URL always present)
    3. ``hpa_url`` always points to the ``.tif`` file: HPA serves every image as
       TIFF at ``<image>.tif`` (302 to the EBI BioStudies mirror) even when the
       XML lists only a JPG. ``image_type`` records what the XML listed.
    4. Remove the secondary tissue categories (Endometrium 2, Stomach 2,
       Soft tissue 2, Skin 2).
    5. Antibody filters: drop
       "uncertain" antibodies; drop antibodies that target more than one gene
       (from ``antibodies.csv``); among antibodies that still have images, keep
       one antibody per gene (enhanced > supported > approved, seeded tie-break).
    6. Require unique ``image_id``. HPA cross-lists "Soft tissue 1" images as
       "Adipose tissue"; ``--duplicate-policy prefer`` (default) keeps the
       "Soft tissue 1" row.
    7. Split by antibody (no antibody in two sets), seeded: two scikit-learn
       ``train_test_split`` stages (test, then val) stratified by antibody source
       (CAB/HPA) x antibody-number quintile, which balances staining era and
       source. Tissue balance follows from every antibody covering (nearly) all
       tissues and is checked with ``check_dataset_splits.py``.

Stage B - pilot subset
    Restrict to the 20 ``PILOT_TISSUES``; per (antibody, tissue) keep 2 of 3
    images; drop one image from 40 % of the 2-image pairs; optional exclusion
    list; re-split as above.

``--split-from`` copies the ``set`` label of an existing CSV: by ``image_id``, then to the other images of
the same antibody, then (if that CSV has ``ensembl_id``) to every antibody of the same gene. Rows without a
match are split by the seeded procedure. The CSV must carry ``split_version`` = ``SPLIT_VERSION``
(``--allow-legacy-split-from`` overrides). Release order: the full dataset inherits from the previous full
dataset, then the pilot inherits from the new full dataset (the first v2 release: pilot fresh, then full
``--split-from`` the new pilot), so pilot and full agree per gene.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("prepare_pilot_dataset")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

IDR_HTTPS_BASE = "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/"

SECONDARY_TISSUES = ["Endometrium 2", "Stomach 2", "Soft tissue 2", "Skin 2"]

# Higher = more reliable (HPA antibody reliability score).
RELIABILITY_RANK = {"enhanced": 3, "supported": 2, "approved": 1}

# Tissue difficulty bins (Behnoush, 2025). The pilot takes 7 easy, 8 moderate
# and all 5 difficult tissues, sampled once with random.seed(42); the result is
# frozen below as PILOT_TISSUES.
TISSUE_DIFFICULTY = {
    "difficult": ["Bone marrow", "Caudate", "Lung", "Lymph node", "Tonsil"],
    "moderate": [
        "Adrenal gland", "Breast", "Cerebellum", "Cerebral cortex", "Endometrium 1",
        "Epididymis", "Hippocampus", "Kidney", "Ovary", "Parathyroid gland", "Placenta",
        "Prostate", "Seminal vesicle", "Soft tissue 1", "Spleen", "Stomach 1", "Testis",
    ],
    "easy": [
        "Appendix", "Bronchus", "Cervix", "Colon", "Duodenum", "Esophagus", "Fallopian tube",
        "Gallbladder", "Heart muscle", "Liver", "Nasopharynx", "Oral mucosa", "Pancreas",
        "Rectum", "Salivary gland", "Skeletal muscle", "Skin 1", "Small intestine",
        "Smooth muscle", "Thyroid gland", "Urinary bladder", "Vagina",
    ],
}
PILOT_TISSUES = [
    "Adrenal gland", "Appendix", "Bone marrow", "Breast", "Caudate", "Cerebellum",
    "Cerebral cortex", "Colon", "Duodenum", "Gallbladder", "Heart muscle", "Hippocampus",
    "Lung", "Lymph node", "Parathyroid gland", "Seminal vesicle", "Small intestine",
    "Testis", "Tonsil", "Urinary bladder",
]
PAIR_DROP_PROB = 0.4  # fraction of 2-image (antibody, tissue) pairs that lose one image

# HPA cross-lists every "Soft tissue 1" image under "Adipose tissue" as well (same image_id,
# same patient). With --duplicate-policy prefer the row whose tissue is listed here wins.
TISSUE_PRECEDENCE = ["Soft tissue 1"]

GROUP_KEY = ["antibody_id", "tissue"]

OUTPUT_COLUMNS = [
    "image_id", "antibody_id", "ensembl_id", "gene_name", "tissue", "organ", "patient_id",
    "image_type", "hpa_url", "idr_available", "idr_ftp_path", "idr_url", "local_path", "set",
    "split_version",
]
OPTIONAL_OUTPUT_COLUMNS = ["set_v1"]   # written only when --set-v1-from is given

# v1: skmultilearn iterative stratification on an ID-sorted antibody list (2025-12 release; val/test skewed to
# old and CAB antibodies). v2: source x antibody-number strata, see split_antibodies().
SPLIT_VERSION = "v2"
SET_NAMES = ("train", "val", "test")
N_ERA_BINS = 5       # antibody-number quantile bins per source (proxy for staining era)
MIN_STRATUM = 10     # antibodies; smaller strata fall back to a coarser one

CSV_READ_KWARGS = dict(dtype=str, keep_default_na=False, low_memory=False)


def read_csv(path):
    """Read a pipeline CSV with every column as string (ids must never become floats)."""
    return pd.read_csv(path, **CSV_READ_KWARGS)


def normalize_patient_id(df):
    """Turn float-formatted ids ("1957.0") back into "1957".

    The Dec-2025 crawl was rewritten through pandas with default dtypes, so almost every
    patient_id carries a trailing ".0". Returns (df, n_changed).
    """
    df = df.copy()
    mask = df["patient_id"].str.fullmatch(r"\d+\.0")
    df.loc[mask, "patient_id"] = df.loc[mask, "patient_id"].str[:-2]
    return df, int(mask.sum())


# --------------------------------------------------------------------------- #
# Step bookkeeping
# --------------------------------------------------------------------------- #

class StepLog:
    """Collects row/antibody/gene/tissue counts after each step."""

    def __init__(self, stage):
        self.stage = stage
        self.rows = []

    def add(self, step, df):
        rec = {
            "stage": self.stage,
            "step": step,
            "images": len(df),
            "antibodies": df["antibody_id"].nunique() if len(df) else 0,
            "genes": df["ensembl_id"].nunique() if len(df) else 0,
            "tissues": df["tissue"].nunique() if len(df) else 0,
        }
        self.rows.append(rec)
        logger.info("[%s] %-40s images=%-9d antibodies=%-6d genes=%-6d tissues=%d",
                    self.stage, step, rec["images"], rec["antibodies"], rec["genes"], rec["tissues"])

    def frame(self):
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------- #
# Stage A helpers
# --------------------------------------------------------------------------- #

def idr_path_mapping(idr_table):
    """image_id -> ftp path (relative to the idr0043 root) from the IDR table."""
    paths = idr_table["ftp_path"]
    paths = paths[paths != ""]
    ids = paths.map(lambda p: os.path.splitext(os.path.basename(p))[0])
    return dict(zip(ids, paths))


def attach_idr(images, idr_table):
    """Add hpa_url / idr_ftp_path / idr_available / idr_url columns.

    ``hpa_url`` is the ``.tif`` form of the crawled URL: HPA serves a TIFF for every
    image at ``<image>.tif`` (redirect to EBI BioStudies), also when the XML only
    offered the JPG (verified 2026-09-06).
    """
    df = images.copy()
    if "hpa_url" not in df.columns:
        df = df.rename(columns={"image_url": "hpa_url"})
    df["hpa_url"] = df["hpa_url"].str.replace(r"\.(jpe?g|tif|tiff)$", ".tif", regex=True)
    mapping = idr_path_mapping(idr_table) if idr_table is not None else {}
    df["idr_ftp_path"] = df["image_id"].map(mapping).fillna("")
    df["idr_available"] = df["idr_ftp_path"] != ""
    df["idr_url"] = np.where(df["idr_available"], IDR_HTTPS_BASE + df["idr_ftp_path"], "")
    return df


def filter_source(df, source):
    if source == "idr":
        return df[df["idr_available"]]
    if source == "all":
        return df
    raise ValueError(f"unknown source {source!r}")


def remove_secondary_tissues(df):
    return df[~df["tissue"].isin(SECONDARY_TISSUES)]


def multi_target_antibodies(antibodies):
    """Antibody ids that map to more than one Ensembl gene in antibodies.csv."""
    n = antibodies.groupby("antibody_id")["ensembl_id"].nunique()
    return set(n[n > 1].index)


def select_best_antibody_per_gene(antibodies, seed):
    """One antibody per gene: highest reliability, seeded random tie-break.

    ``antibodies`` must already be restricted to the candidate set.
    """
    ab = antibodies.copy()
    unknown = set(ab["Reliability score"]) - set(RELIABILITY_RANK)
    if unknown:
        raise ValueError(f"Unknown reliability scores: {sorted(unknown)}")
    ab["rel_rank"] = ab["Reliability score"].map(RELIABILITY_RANK)
    ab = ab.sample(frac=1, random_state=seed).reset_index(drop=True)
    ab = ab.sort_values(["ensembl_id", "rel_rank"], ascending=[True, False], kind="mergesort")
    return ab.drop_duplicates("ensembl_id", keep="first").drop(columns="rel_rank")


def filter_antibodies(images, antibodies, seed, steps):
    """Antibody filters: uncertain -> multi-target (from antibodies.csv) -> one antibody per gene
    chosen among antibodies that still have images."""
    ab = antibodies[antibodies["Reliability score"] != "uncertain"]
    images = images[images["antibody_id"].isin(ab["antibody_id"])]
    steps.add("remove uncertain antibodies", images)

    multi = multi_target_antibodies(antibodies)
    ab = ab[~ab["antibody_id"].isin(multi)]
    images = images[~images["antibody_id"].isin(multi)]
    steps.add("remove multi-target antibodies", images)

    candidates = ab[ab["antibody_id"].isin(images["antibody_id"].unique())]
    best = select_best_antibody_per_gene(candidates, seed)
    images = images[images["antibody_id"].isin(best["antibody_id"])]
    steps.add("one antibody per gene", images)
    return images


def resolve_duplicates(df, policy, report_path):
    """Enforce unique image_id.

    policy:
      fail   - write the duplicated rows to report_path and exit 2
      prefer - for ids listed under several tissues keep the row whose tissue is in
               TISSUE_PRECEDENCE (exactly one such row); ids that cannot be resolved
               that way lose all their rows
      drop   - drop all rows of every duplicated id
    """
    dup_mask = df.duplicated("image_id", keep=False)
    n_dup_ids = df.loc[dup_mask, "image_id"].nunique()
    if n_dup_ids == 0:
        return df
    dups = df[dup_mask].sort_values(["image_id", "tissue"])
    pairs = (dups.groupby("image_id")["tissue"].agg(lambda t: " | ".join(sorted(t)))
             .value_counts().head(5).to_dict())
    logger.warning("%d image_ids appear more than once (%d rows); tissue pairs: %s",
                   n_dup_ids, len(dups), pairs)
    if policy == "fail":
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        dups.to_csv(report_path, index=False)
        logger.error("Duplicate image_ids present; rows written to %s. Re-run with --duplicate-policy "
                     "prefer|drop, or fix the crawl.", report_path)
        raise SystemExit(2)
    if policy == "prefer":
        preferred = dups[dups["tissue"].isin(TISSUE_PRECEDENCE)]
        per_id = preferred.groupby("image_id").size()
        resolved = set(per_id[per_id == 1].index)
        keep = dups["image_id"].isin(resolved) & dups["tissue"].isin(TISSUE_PRECEDENCE)
        drop_idx = dups.index[~keep]
        logger.info("prefer %s: %d ids resolved, %d ids dropped entirely",
                    TISSUE_PRECEDENCE, len(resolved), n_dup_ids - len(resolved))
        return df.drop(index=drop_idx)
    if policy == "drop":
        return df[~dup_mask]
    raise ValueError(f"unknown duplicate policy {policy!r}")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def antibody_strata(antibody_ids, n_bins=N_ERA_BINS, min_size=MIN_STRATUM):
    """Stratum label per antibody: source prefix (CAB, HPA, ...) x antibody-number quantile bin within the source.

    HPA antibody numbers grow with the staining date (Spearman ~0.8 in the 2025-12 data) and CAB antibodies are a
    separate, older series, so these strata balance staining era and antibody source across the sets. Ids that do
    not look like ``<letters><digits>`` form the stratum ``other``. Strata with fewer than ``min_size`` antibodies
    fall back to their source, then to a common stratum, then into the largest stratum.
    """
    ids = pd.Series(list(antibody_ids), dtype=str)
    parts = ids.str.extract(r"^([A-Za-z]+)(\d+)$")
    source = parts[0].fillna("other")
    number = pd.to_numeric(parts[1], errors="coerce")
    label = source.copy()
    for src, idx in source.groupby(source).groups.items():
        if src == "other":
            continue
        k = min(n_bins, max(1, len(idx) // min_size))
        bins = pd.qcut(number.loc[idx].rank(method="first"), k, labels=False).astype(int)
        label.loc[idx] = src + "_" + bins.astype(str)
    for coarser in (source, pd.Series("all", index=ids.index)):
        small = label.map(label.value_counts()) < min_size
        label = label.where(~small, coarser)
    small = label.map(label.value_counts()) < min_size
    if small.any() and (~small).any():
        label[small] = label[~small].value_counts().idxmax()
    return label.to_numpy()


def _split_off(ids, strata, size, seed):
    """Hold out ``size`` antibodies (seeded train_test_split), stratified unless a stratum is too small."""
    if size <= 0:
        return ids, ids[:0], strata
    stratify = strata if len(set(strata)) > 1 else None
    try:
        keep, held, s_keep, _ = train_test_split(ids, strata, test_size=size, random_state=seed, stratify=stratify)
    except ValueError as err:
        logger.warning("stratified split of %d antibodies not possible (%s); splitting without strata", len(ids), err)
        keep, held, s_keep, _ = train_test_split(ids, strata, test_size=size, random_state=seed)
    return keep, held, s_keep


def split_antibodies(df, train_frac, val_frac, test_frac, seed):
    """antibody_id -> set for every antibody in df (split version v2).

    Two seeded scikit-learn ``train_test_split`` stages, test first, then val from the rest (sizes rounded from
    ``test_frac`` and ``val_frac`` of all antibodies), both stratified by
    :func:`antibody_strata` (source x antibody-number quintile). The antibody list is sorted first, so the result
    depends only on the set of antibodies and the seed. Tissue is not a stratification variable: almost every
    antibody has images in (nearly) all tissues, so any antibody-level split is tissue-balanced;
    ``check_dataset_splits.py`` verifies this on the real CSVs. Fewer than 3 antibodies all go to train.
    """
    ids = np.array(sorted(df["antibody_id"].unique()))
    n = len(ids)
    if n == 0:
        return {}
    if n < 3:
        return {a: "train" for a in ids}
    strata = antibody_strata(ids)
    n_test, n_val = int(round(test_frac * n)), int(round(val_frac * n))   # sizes from the overall fractions
    trainval, test, s_trainval = _split_off(ids, strata, n_test, seed)
    train, val, _ = _split_off(trainval, s_trainval, min(n_val, len(trainval) - 1), seed)
    assign = {a: "train" for a in train}
    assign.update({a: "val" for a in val})
    assign.update({a: "test" for a in test})
    return assign


def check_split_from_version(path, allow_legacy=False):
    """Refuse a --split-from CSV that was not made by the current split version (it would re-import its bias)."""
    head = pd.read_csv(path, nrows=0)
    if "split_version" in head.columns:
        versions = set(pd.read_csv(path, usecols=["split_version"], dtype=str,
                                   keep_default_na=False)["split_version"])
    else:
        versions = set()
    if versions == {SPLIT_VERSION}:
        return
    msg = (f"--split-from {path} has split_version {sorted(versions) or 'missing'}, expected {SPLIT_VERSION!r}; "
           f"inheriting an older split keeps its train/val/test imbalance")
    if not allow_legacy:
        raise ValueError(msg + " (pass --allow-legacy-split-from to inherit it anyway)")
    logger.warning(msg)


def assign_sets(df, train_frac, val_frac, test_frac, seed, split_from=None):
    """Add 'set' and 'split_version' columns. Optionally inherit labels from an existing dataset CSV.

    Inheritance from ``split_from``, in this order: by ``image_id``; then every other image of an antibody that
    has an inherited image; then, if that CSV has ``ensembl_id``, every antibody of a gene listed there (the pilot
    and the full dataset can use different antibodies for one gene). Aborts if the CSV gives one antibody or one
    gene several sets. Antibodies without any match are split by :func:`split_antibodies`.
    """
    df = df.copy()
    df["set"] = ""
    df["split_version"] = SPLIT_VERSION
    if split_from is not None:
        prev = read_csv(split_from)
        if "set" not in prev.columns:
            raise ValueError(f"{split_from} has no 'set' column")
        prev = prev[prev["set"].isin(SET_NAMES)]
        multi = prev.groupby("image_id")["set"].nunique()
        if (multi > 1).any():
            raise RuntimeError(f"{int((multi > 1).sum())} image_ids have several sets in {split_from}")
        prev = prev.drop_duplicates("image_id")
        df["set"] = df["image_id"].map(prev.set_index("image_id")["set"]).fillna("")
        by_image = int((df["set"] != "").sum())

        labeled = df[df["set"] != ""]
        per_ab_n = labeled.groupby("antibody_id")["set"].nunique()
        if (per_ab_n > 1).any():
            bad = per_ab_n[per_ab_n > 1].index[:5].tolist()
            raise RuntimeError(f"{int((per_ab_n > 1).sum())} antibodies have images in several sets in "
                               f"{split_from} (e.g. {bad}); fix that file or drop --split-from")
        per_ab = labeled.groupby("antibody_id")["set"].first()
        fill_ab = (df["set"] == "") & df["antibody_id"].isin(per_ab.index)
        df.loc[fill_ab, "set"] = df.loc[fill_ab, "antibody_id"].map(per_ab)

        by_gene = 0
        if "ensembl_id" in prev.columns:
            if (prev["ensembl_id"] == "").any() or (df["ensembl_id"] == "").any():
                raise ValueError("empty ensembl_id: gene-level --split-from needs a gene for every row")
            per_gene_n = prev.groupby("ensembl_id")["set"].nunique()
            if (per_gene_n > 1).any():
                bad = per_gene_n[per_gene_n > 1].index[:5].tolist()
                raise RuntimeError(f"{int((per_gene_n > 1).sum())} genes span several sets in {split_from} "
                                   f"(e.g. {bad})")
            per_gene = prev.groupby("ensembl_id")["set"].first()
            fill_gene = (df["set"] == "") & df["ensembl_id"].isin(per_gene.index)
            df.loc[fill_gene, "set"] = df.loc[fill_gene, "ensembl_id"].map(per_gene)
            by_gene = int(fill_gene.sum())
        else:
            logger.info("split-from %s has no ensembl_id column; gene-level inheritance skipped",
                        os.path.basename(split_from))
        logger.info("split-from %s: %d rows by image_id, %d by antibody, %d by gene",
                    os.path.basename(split_from), by_image, int(fill_ab.sum()), by_gene)

    todo = df[df["set"] == ""]
    if len(todo):
        assign = split_antibodies(todo, train_frac, val_frac, test_frac, seed)
        df.loc[df["set"] == "", "set"] = todo["antibody_id"].map(assign)
        logger.info("seeded split assigned %d rows (%d antibodies)", len(todo), len(assign))

    spans = df.groupby("antibody_id")["set"].nunique()
    if (spans > 1).any():
        raise RuntimeError(f"{int((spans > 1).sum())} antibodies span several sets")
    for name in SET_NAMES:
        n = int((df["set"] == name).sum())
        logger.info("  %-5s %9d images (%.1f%%)", name, n, 100.0 * n / max(len(df), 1))
    return df


def attach_set_v1(df, path):
    """Add 'set_v1': the 'set' label of the same image_id in an older (v1) dataset CSV, empty if absent."""
    prev = read_csv(path)
    prev = prev[prev["set"].isin(SET_NAMES)].drop_duplicates("image_id")
    df = df.copy()
    df["set_v1"] = df["image_id"].map(prev.set_index("image_id")["set"]).fillna("")
    logger.info("set_v1 from %s: %d of %d rows labelled", os.path.basename(path),
                int((df["set_v1"] != "").sum()), len(df))
    return df


# --------------------------------------------------------------------------- #
# Stage B helpers (pilot thinning)
# --------------------------------------------------------------------------- #

def _shuffled_group_stats(df, seed):
    shuffled = df.sample(frac=1, random_state=seed)
    grp = shuffled.groupby(GROUP_KEY, sort=False)
    size = grp["image_id"].transform("size")
    rank = grp.cumcount()
    gid = grp.ngroup()
    return shuffled, size, rank, gid


def thin_triplets(df, seed):
    """For each (antibody, tissue) with exactly 3 images keep 2 (seeded)."""
    shuffled, size, rank, _ = _shuffled_group_stats(df, seed)
    drop_idx = shuffled.index[(size == 3) & (rank == 2)]
    return df.drop(index=drop_idx)


def thin_pairs(df, seed, drop_prob=PAIR_DROP_PROB):
    """For each (antibody, tissue) with exactly 2 images drop one with probability drop_prob (seeded)."""
    shuffled, size, rank, gid = _shuffled_group_stats(df, seed)
    pair_groups = np.unique(gid[size == 2])
    rng = np.random.default_rng(seed)
    doomed = set(pair_groups[rng.random(len(pair_groups)) < drop_prob])
    drop_idx = shuffled.index[(size == 2) & (rank == 1) & gid.isin(doomed)]
    return df.drop(index=drop_idx)


def sample_pilot_tissues(seed=42, n_easy=7, n_moderate=8):
    """Regenerate PILOT_TISSUES from the difficulty bins (documentation / test helper)."""
    import random
    rng = random.Random(seed)
    easy = rng.sample(TISSUE_DIFFICULTY["easy"], n_easy)
    moderate = rng.sample(TISSUE_DIFFICULTY["moderate"], n_moderate)
    return sorted(easy + moderate + TISSUE_DIFFICULTY["difficult"])


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def build_full_dataset(images, antibodies, idr_table, source, seed,
                       duplicate_policy="prefer", duplicate_report="duplicate_image_ids.csv",
                       fracs=(0.70, 0.10, 0.20), split_from=None, steps=None):
    steps = steps if steps is not None else StepLog("full")
    df, n_fixed = normalize_patient_id(images)
    if n_fixed:
        logger.info("normalized %d float-formatted patient_id values", n_fixed)
    df = attach_idr(df, idr_table)
    steps.add("crawled images", df)
    df = filter_source(df, source)
    steps.add(f"source = {source}", df)
    df = remove_secondary_tissues(df)
    steps.add("remove secondary tissue categories", df)
    df = filter_antibodies(df, antibodies, seed, steps)
    df = resolve_duplicates(df, duplicate_policy, duplicate_report)
    steps.add("unique image_id", df)
    df = assign_sets(df, *fracs, seed=seed, split_from=split_from)
    return df


def build_pilot_subset(df, seed, exclude_ids=None, fracs=(0.70, 0.10, 0.20),
                       split_from=None, steps=None):
    """Pilot subset: pilot tissues, thinning, optional exclusions, re-split."""
    steps = steps if steps is not None else StepLog("pilot")
    df = df[df["tissue"].isin(PILOT_TISSUES)]
    steps.add(f"restrict to {len(PILOT_TISSUES)} pilot tissues", df)
    df = thin_triplets(df, seed)
    steps.add("keep 2 of 3 images per antibody x tissue", df)
    df = thin_pairs(df, seed)
    steps.add(f"drop one image from {int(PAIR_DROP_PROB * 100)}% of 2-image pairs", df)
    if exclude_ids:
        df = df[~df["image_id"].isin(set(exclude_ids))]
        steps.add("exclude listed image_ids", df)
    df = df.drop(columns=["set"], errors="ignore")
    df = assign_sets(df, *fracs, seed=seed, split_from=split_from)
    return df


def finalize_columns(df):
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"output is missing columns {missing}")
    cols = OUTPUT_COLUMNS + [c for c in OPTIONAL_OUTPUT_COLUMNS if c in df.columns]
    return df[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def save_distribution_plots(df, output_dir, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    for col, figsize in (("tissue", (12, 4)), ("organ", (12, 6))):
        counts = df[col].value_counts()
        fig, ax = plt.subplots(figsize=figsize)
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Distribution of {col}s ({prefix})")
        ax.set_xlabel(col.capitalize())
        ax.set_ylabel("Number of images")
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_{col}_distribution.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("saved %s", path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def default_paths():
    scratch = os.environ.get("SCRATCH", os.path.expanduser("~/scratch"))
    meta = os.path.join(scratch, "hpa_tissue", "metadata")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return {
        "images": os.path.join(meta, "images.csv"),
        "antibodies": os.path.join(meta, "antibodies.csv"),
        "idr_table": os.path.join(repo, "data", "external",
                                  "idr0043-experimentA-annotation-M-00100_ftp_paths.csv"),
        "out_dir": meta,
    }


def parse_args(argv=None):
    d = default_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-csv", default=d["images"], help="Crawled image table from hpa_xml_parser.py")
    p.add_argument("--antibodies-csv", default=d["antibodies"], help="Antibody table from hpa_xml_parser.py")
    p.add_argument("--idr-table", default=d["idr_table"],
                   help="IDR normal-tissue table with ftp_path (build_idr_ftp_paths.py). "
                        "Required for --source idr; optional otherwise.")
    p.add_argument("--source", choices=["idr", "all"], default="idr",
                   help="idr: only images mirrored on IDR; all: every crawled image")
    p.add_argument("--stage", choices=["full", "pilot", "both"], default="both")
    p.add_argument("--full-output", default=None, help="Default: <out-dir>/HPA_full_dataset.csv")
    p.add_argument("--pilot-output", default=None, help="Default: <out-dir>/HPA_pilot_dataset.csv")
    p.add_argument("--out-dir", default=d["out_dir"])
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-from", default=None,
                   help="Existing dataset CSV (same split_version); its 'set' labels are inherited by image_id, "
                        "antibody and, if it has ensembl_id, gene")
    p.add_argument("--allow-legacy-split-from", action="store_true",
                   help=f"Allow a --split-from CSV whose split_version is not {SPLIT_VERSION!r}")
    p.add_argument("--set-v1-from", default=None,
                   help="Older (v1) dataset CSV; its 'set' label is written to a 'set_v1' column by image_id. "
                        "Use with --stage pilot or --stage full (v1 pilot and full labels differ)")
    p.add_argument("--exclude-ids", default=None,
                   help="CSV with an image_id column; those images are dropped from the pilot subset")
    p.add_argument("--duplicate-policy", choices=["fail", "prefer", "drop"], default="prefer",
                   help="image_id listed under several tissues: fail (exit 2), prefer (keep the "
                        f"{TISSUE_PRECEDENCE} row, drop unresolved ids), drop (drop all rows of those ids)")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plot-dir", default=None, help="Default: <out-dir>")
    p.add_argument("--log-file", default=None)
    args = p.parse_args(argv)

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        p.error("train/val/test fractions must sum to 1")
    if args.source == "idr" and not os.path.exists(args.idr_table):
        p.error(f"--source idr needs --idr-table (not found: {args.idr_table})")
    args.full_output = args.full_output or os.path.join(args.out_dir, "HPA_full_dataset.csv")
    args.pilot_output = args.pilot_output or os.path.join(args.out_dir, "HPA_pilot_dataset.csv")
    args.plot_dir = args.plot_dir or args.out_dir
    return args


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_file)
    fracs = (args.train_frac, args.val_frac, args.test_frac)
    logger.info("images=%s antibodies=%s idr_table=%s source=%s seed=%d",
                args.images_csv, args.antibodies_csv, args.idr_table, args.source, args.seed)

    if args.split_from:
        check_split_from_version(args.split_from, args.allow_legacy_split_from)
    images = read_csv(args.images_csv)
    antibodies = read_csv(args.antibodies_csv)
    idr_table = read_csv(args.idr_table) if os.path.exists(args.idr_table) else None
    if idr_table is None:
        logger.warning("IDR table not found; idr_* columns will be empty")

    steps = StepLog("full")
    full = build_full_dataset(
        images, antibodies, idr_table, args.source, args.seed,
        duplicate_policy=args.duplicate_policy,
        duplicate_report=os.path.join(args.out_dir, "duplicate_image_ids.csv"),
        fracs=fracs, split_from=args.split_from, steps=steps,
    )
    if args.set_v1_from and args.stage in ("full", "both"):
        full = attach_set_v1(full, args.set_v1_from)
    full_out = finalize_columns(full)
    os.makedirs(os.path.dirname(os.path.abspath(args.full_output)), exist_ok=True)
    if args.stage in ("full", "both"):
        full_out.to_csv(args.full_output, index=False)
        logger.info("wrote %s (%d rows)", args.full_output, len(full_out))
        if args.save_plots:
            save_distribution_plots(full_out, args.plot_dir, "full")

    if args.stage in ("pilot", "both"):
        exclude = None
        if args.exclude_ids:
            exclude = read_csv(args.exclude_ids)["image_id"].tolist()
        pilot_steps = StepLog("pilot")
        pilot = build_pilot_subset(full, args.seed, exclude_ids=exclude, fracs=fracs,
                                   split_from=args.split_from, steps=pilot_steps)
        if args.set_v1_from:
            pilot = attach_set_v1(pilot, args.set_v1_from)
        pilot_out = finalize_columns(pilot)
        pilot_out.to_csv(args.pilot_output, index=False)
        logger.info("wrote %s (%d rows)", args.pilot_output, len(pilot_out))
        steps.rows.extend(pilot_steps.rows)
        if args.save_plots:
            save_distribution_plots(pilot_out, args.plot_dir, "pilot")

    logger.info("rows after each step:\n%s", steps.frame().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
