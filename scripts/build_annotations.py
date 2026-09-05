#!/usr/bin/env python3
"""
Attach the HPA annotations to a dataset CSV.

HPA annotates staining per antibody x tissue x cell type (not per image); patient
sex/age per antibody x tissue x patient; tissue ontology terms per antibody x
tissue. ``hpa_xml_parser.py`` writes these as ``cells.csv``, ``patients.csv`` and
``tissues.csv``. This script joins them onto a dataset CSV from
``prepare_pilot_dataset.py`` and writes three files next to it:

``<prefix>_cell_annotations.csv``   one row per image x annotated cell type:
    staining / intensity / quantity / location as HPA strings plus ordinal codes
    (``*_code``: 0..3, -1 = missing or "not representative").
``<prefix>_image_annotations.csv``  one row per image: patient sex/age, tissue
    ontology terms, the list of annotated cell types and the maximum codes over
    the cell types (``protein_detected`` = any cell type stained).
``<prefix>_cell_types.csv``         cell-type vocabulary with image/antibody counts.

Example:
    python build_annotations.py --dataset-csv HPA_pilot_dataset_idr.csv \\
        --metadata-dir ./hpa_tissue/metadata --out-dir ./hpa_tissue/metadata
"""

import argparse
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger("build_annotations")

CSV_READ_KWARGS = dict(dtype=str, keep_default_na=False, low_memory=False)
SAMPLE_KEY = ["antibody_id", "ensembl_id", "tissue"]

STAINING_CODE = {"not detected": 0, "low": 1, "medium": 2, "high": 3}
INTENSITY_CODE = {"negative": 0, "weak": 1, "moderate": 2, "strong": 3}
QUANTITY_CODE = {"none": 0, "<25%": 1, "25%-75%": 2, "25-75%": 2, "25% - 75%": 2, ">75%": 3}
LOCATION_CODE = {"none": 0, "cytoplasmic/membranous": 1, "nuclear": 2, "both": 3}

DATASET_COLUMNS = ["image_id", "antibody_id", "ensembl_id", "gene_name", "tissue", "organ", "patient_id"]


def code(series, mapping):
    """Map HPA strings to ordinal codes; anything unknown (empty, 'not representative') -> -1."""
    return series.str.strip().str.lower().map({k.lower(): v for k, v in mapping.items()}).fillna(-1).astype(int)


def location_code(series):
    s = series.str.lower()
    has_cm = s.str.contains("cytoplasmic/membranous", regex=False)
    has_nuc = s.str.contains("nuclear", regex=False)
    out = pd.Series(-1, index=series.index)
    out[s.str.strip() == "none"] = LOCATION_CODE["none"]
    out[has_cm & ~has_nuc] = LOCATION_CODE["cytoplasmic/membranous"]
    out[~has_cm & has_nuc] = LOCATION_CODE["nuclear"]
    out[has_cm & has_nuc] = LOCATION_CODE["both"]
    return out


def dedupe(df, key, what):
    """Drop identical rows; for keys that still repeat with different values keep the first."""
    before = len(df)
    df = df.drop_duplicates()
    conflicts = int(df.duplicated(key, keep=False).sum())
    if conflicts:
        logger.warning("%s: %d rows share a key with different values; keeping the first", what, conflicts)
    df = df.drop_duplicates(key, keep="first")
    logger.info("%s: %d rows -> %d unique", what, before, len(df))
    return df


def build_cell_annotations(dataset, cells):
    cells = dedupe(cells, SAMPLE_KEY + ["cell_type"], "cells.csv")
    out = dataset.merge(cells, on=SAMPLE_KEY, how="inner")
    out["staining_code"] = code(out["staining"], STAINING_CODE)
    out["intensity_code"] = code(out["intensity"], INTENSITY_CODE)
    out["quantity_code"] = code(out["quantity"], QUANTITY_CODE)
    out["location_code"] = location_code(out["location"])
    cols = [c for c in dataset.columns] + ["cell_type", "staining", "intensity", "quantity", "location",
                                           "staining_code", "intensity_code", "quantity_code", "location_code"]
    return out[cols].sort_values(["image_id", "cell_type"]).reset_index(drop=True)


def build_image_annotations(dataset, cell_annotations, patients, tissues):
    img = dataset.copy()
    if patients is not None:
        pat = dedupe(patients[SAMPLE_KEY + ["patient_id", "sex", "age"]], SAMPLE_KEY + ["patient_id"], "patients.csv")
        img = img.merge(pat, on=SAMPLE_KEY + ["patient_id"], how="left")
    if tissues is not None:
        ont = (tissues.drop_duplicates()
               .groupby(SAMPLE_KEY)["ontology_terms"]
               .agg(lambda s: ";".join(sorted(set(x for x in s if x)))).reset_index())
        img = img.merge(ont, on=SAMPLE_KEY, how="left")
    agg = cell_annotations.groupby("image_id").agg(
        n_cell_types=("cell_type", "size"),
        cell_types=("cell_type", lambda s: ";".join(sorted(s))),
        max_staining_code=("staining_code", "max"),
        max_intensity_code=("intensity_code", "max"),
        max_quantity_code=("quantity_code", "max"),
    ).reset_index()
    img = img.merge(agg, on="image_id", how="left")
    img["n_cell_types"] = img["n_cell_types"].fillna(0).astype(int)
    for c in ("max_staining_code", "max_intensity_code", "max_quantity_code"):
        img[c] = img[c].fillna(-1).astype(int)
    img["cell_types"] = img["cell_types"].fillna("")
    img["protein_detected"] = img["max_staining_code"] > 0
    return img.fillna("").sort_values("image_id").reset_index(drop=True)


def build_cell_type_vocabulary(cell_annotations):
    voc = cell_annotations.groupby("cell_type").agg(
        n_images=("image_id", "nunique"), n_antibodies=("antibody_id", "nunique"),
        n_tissues=("tissue", "nunique"), tissues=("tissue", lambda s: ";".join(sorted(set(s)))),
    ).reset_index().sort_values("n_images", ascending=False)
    return voc.reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-csv", required=True, help="Dataset CSV from prepare_pilot_dataset.py")
    p.add_argument("--metadata-dir", required=True,
                   help="Directory with cells.csv, patients.csv, tissues.csv from hpa_xml_parser.py")
    p.add_argument("--out-dir", default=None, help="Output directory (default: directory of --dataset-csv)")
    p.add_argument("--prefix", default=None, help="Output file prefix (default: dataset file name without .csv[.gz])")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.dataset_csv))
    prefix = args.prefix or os.path.basename(args.dataset_csv).replace(".csv.gz", "").replace(".csv", "")
    os.makedirs(out_dir, exist_ok=True)

    dataset = pd.read_csv(args.dataset_csv, **CSV_READ_KWARGS)
    missing = [c for c in DATASET_COLUMNS if c not in dataset.columns]
    if missing:
        p.error(f"dataset CSV lacks columns {missing}")
    keep = DATASET_COLUMNS + (["set"] if "set" in dataset.columns else [])
    dataset = dataset[keep]

    def load(name, required):
        path = os.path.join(args.metadata_dir, name)
        if not os.path.exists(path):
            if required:
                p.error(f"{path} not found")
            logger.warning("%s not found; skipping", path)
            return None
        return pd.read_csv(path, **CSV_READ_KWARGS)

    cells = load("cells.csv", required=True)
    patients = load("patients.csv", required=False)
    tissues = load("tissues.csv", required=False)

    cell_ann = build_cell_annotations(dataset, cells)
    img_ann = build_image_annotations(dataset, cell_ann, patients, tissues)
    vocab = build_cell_type_vocabulary(cell_ann)

    n_without = int((img_ann["n_cell_types"] == 0).sum())
    logger.info("%d images, %d image x cell-type rows, %d cell types, %d images without any cell annotation",
                len(img_ann), len(cell_ann), len(vocab), n_without)
    for name, df in (("cell_annotations", cell_ann), ("image_annotations", img_ann), ("cell_types", vocab)):
        path = os.path.join(out_dir, f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
