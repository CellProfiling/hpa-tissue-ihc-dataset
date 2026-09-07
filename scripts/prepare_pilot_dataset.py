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
    7. Split by antibody (no antibody in two sets), stratified by tissue, seeded.

Stage B - pilot subset
    Restrict to the 20 ``PILOT_TISSUES``; per (antibody, tissue) keep 2 of 3
    images; drop one image from 40 % of the 2-image pairs; optional exclusion
    list; re-split as above.

``--split-from`` copies the ``set`` label of an existing CSV by ``image_id`` so
a re-generation keeps earlier train/val/test membership. Rows without a match are split by the seeded
procedure.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification

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
]

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

def strat_split(X, y, test_size, seed):
    """Seeded iterative stratification (body of skmultilearn's iterative_train_test_split + seed).

    skmultilearn 0.2.0 breaks ties with the *global* ``np.random`` and ignores its
    own ``random_state`` argument (passing it even raises with sklearn >= 1.3), so
    the global RNG is seeded for the duration of the split and restored afterwards.
    """
    strat = IterativeStratification(
        n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0 - test_size],
    )
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        train_idx, test_idx = next(strat.split(X, y))
    finally:
        np.random.set_state(state)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def split_antibodies(df, train_frac, val_frac, test_frac, seed):
    """antibody_id -> set for every antibody in df (tissue-stratified, seeded)."""
    grouped = df.groupby("antibody_id")["tissue"].unique().reset_index()
    n = len(grouped)
    if n == 0:
        return {}
    if n < 3:
        return {a: "train" for a in grouped["antibody_id"]}
    X = np.arange(n).reshape(-1, 1)
    y = MultiLabelBinarizer().fit_transform(grouped["tissue"])
    X_trainval, y_trainval, X_test, _ = strat_split(X, y, test_frac, seed)
    val_rel = val_frac / (train_frac + val_frac)
    X_train, _, X_val, _ = strat_split(X_trainval, y_trainval, val_rel, seed)
    assign = {}
    for X_part, name in ((X_train, "train"), (X_val, "val"), (X_test, "test")):
        for i in X_part.flatten():
            assign[grouped.loc[i, "antibody_id"]] = name
    return assign


def assign_sets(df, train_frac, val_frac, test_frac, seed, split_from=None):
    """Add a 'set' column. Optionally inherit labels from an existing CSV by image_id."""
    df = df.copy()
    df["set"] = ""
    if split_from is not None:
        prev = read_csv(split_from)
        prev = prev[prev["set"].isin(["train", "val", "test"])].drop_duplicates("image_id")
        df["set"] = df["image_id"].map(prev.set_index("image_id")["set"]).fillna("")
        inherited = int((df["set"] != "").sum())
        # Antibodies with some inherited rows: give the remaining rows the same set
        # (majority vote) so no antibody ends up in two sets.
        labeled = df[df["set"] != ""]
        per_ab = labeled.groupby("antibody_id")["set"].agg(lambda s: s.value_counts().idxmax())
        n_conflict = int((labeled.groupby("antibody_id")["set"].nunique() > 1).sum())
        fill = (df["set"] == "") & df["antibody_id"].isin(per_ab.index)
        df.loc[fill, "set"] = df.loc[fill, "antibody_id"].map(per_ab)
        logger.info("split-from %s: %d rows inherited, %d rows filled from their antibody, "
                    "%d antibodies had conflicting inherited sets (majority kept)",
                    os.path.basename(split_from), inherited, int(fill.sum()), n_conflict)

    todo = df[df["set"] == ""]
    if len(todo):
        assign = split_antibodies(todo, train_frac, val_frac, test_frac, seed)
        df.loc[df["set"] == "", "set"] = todo["antibody_id"].map(assign)
        logger.info("seeded split assigned %d rows (%d antibodies)", len(todo), len(assign))

    spans = df.groupby("antibody_id")["set"].nunique()
    if (spans > 1).any():
        raise RuntimeError(f"{int((spans > 1).sum())} antibodies span several sets")
    for name in ("train", "val", "test"):
        n = int((df["set"] == name).sum())
        logger.info("  %-5s %9d images (%.1f%%)", name, n, 100.0 * n / max(len(df), 1))
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
                       duplicate_policy="prefer", duplicate_report="duplicates.csv",
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
    return df[OUTPUT_COLUMNS].reset_index(drop=True)


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
    p.add_argument("--full-output", default=None, help="Default: <out-dir>/HPA_full_dataset_<source>.csv")
    p.add_argument("--pilot-output", default=None, help="Default: <out-dir>/HPA_pilot_dataset_<source>.csv")
    p.add_argument("--out-dir", default=d["out_dir"])
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-from", default=None,
                   help="Existing dataset CSV; its 'set' labels are inherited by image_id")
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
    args.full_output = args.full_output or os.path.join(args.out_dir, f"HPA_full_dataset_{args.source}.csv")
    args.pilot_output = args.pilot_output or os.path.join(args.out_dir, f"HPA_pilot_dataset_{args.source}.csv")
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
