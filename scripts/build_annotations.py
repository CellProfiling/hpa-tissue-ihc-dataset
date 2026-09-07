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

With ``--subcellular-tsv`` the HPA subcellular location annotation (per gene, from
immunofluorescence of cell lines; ``subcellular_location.tsv.zip`` on
https://www.proteinatlas.org/about/download, downloaded automatically if the path
does not exist) is joined onto the image annotations by ``ensembl_id`` as
``subcellular_*`` columns, and a fourth file is written:

``<prefix>_subcellular_locations.csv``  location vocabulary with gene/image counts.

Example:
    python build_annotations.py --dataset-csv HPA_pilot_dataset.csv \\
        --metadata-dir ./hpa_tissue/metadata --out-dir ./hpa_tissue/metadata \\
        --subcellular-tsv ./hpa_tissue/metadata/subcellular_location.tsv.zip
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

SUBCELLULAR_URL = "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
# HPA column -> output column. "Gene" (Ensembl id) is the join key; "Gene name" is already in the dataset.
# "Reliability" is HPA's summary score per gene; Enhanced/Supported/Approved/Uncertain list the locations
# annotated at that reliability level. Extracellular location, single-cell variation, cell cycle dependency
# and GO ids are left out.
SUBCELLULAR_COLUMNS = {
    "Reliability": "subcellular_reliability",
    "Main location": "subcellular_main_location",
    "Additional location": "subcellular_additional_location",
    "Enhanced": "subcellular_enhanced",
    "Supported": "subcellular_supported",
    "Approved": "subcellular_approved",
    "Uncertain": "subcellular_uncertain",
}


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


def fetch_subcellular_location(path):
    """Download HPA's subcellular_location.tsv.zip to ``path`` unless it already exists."""
    if os.path.exists(path):
        return path
    import requests
    logger.info("downloading %s -> %s", SUBCELLULAR_URL, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    r = requests.get(SUBCELLULAR_URL, timeout=300)
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, path)
    return path


def load_subcellular(path):
    """Read subcellular_location.tsv(.zip): one row per gene, columns renamed to ``subcellular_*``."""
    df = pd.read_csv(path, sep="\t", **CSV_READ_KWARGS)
    missing = [c for c in ["Gene"] + list(SUBCELLULAR_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; expected the HPA subcellular_location.tsv")
    df = df.rename(columns={"Gene": "ensembl_id", **SUBCELLULAR_COLUMNS})
    return dedupe(df[["ensembl_id"] + list(SUBCELLULAR_COLUMNS.values())], ["ensembl_id"], os.path.basename(path))


def attach_subcellular(image_annotations, subcellular):
    """Left-join the per-gene subcellular columns by ensembl_id (empty where the gene has no entry)."""
    out = image_annotations.merge(subcellular, on="ensembl_id", how="left").fillna("")
    covered = out["subcellular_main_location"] != ""
    logger.info("subcellular location: %d of %d images (%d of %d genes) have an entry",
                int(covered.sum()), len(out), out.loc[covered, "ensembl_id"].nunique(), out["ensembl_id"].nunique())
    return out


def build_subcellular_vocabulary(image_annotations):
    """One row per location: number of genes / images with it as main location, and as main or additional."""
    def explode(col):
        s = image_annotations[["ensembl_id", "image_id", col]]
        s = s[s[col] != ""].assign(location=lambda d: d[col].str.split(";")).explode("location")
        return s[["ensembl_id", "image_id", "location"]]
    main = explode("subcellular_main_location")
    anyloc = pd.concat([main, explode("subcellular_additional_location")])
    voc = main.groupby("location").agg(n_genes_main=("ensembl_id", "nunique"), n_images_main=("image_id", "nunique"))
    voc = voc.join(anyloc.groupby("location").agg(n_genes_any=("ensembl_id", "nunique"), n_images_any=("image_id", "nunique")), how="outer")
    voc = voc.fillna(0).astype(int).reset_index().sort_values(["n_images_main", "location"], ascending=[False, True])
    return voc[["location", "n_genes_main", "n_genes_any", "n_images_main", "n_images_any"]].reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-csv", required=True, help="Dataset CSV from prepare_pilot_dataset.py")
    p.add_argument("--metadata-dir", required=True,
                   help="Directory with cells.csv, patients.csv, tissues.csv from hpa_xml_parser.py")
    p.add_argument("--out-dir", default=None, help="Output directory (default: directory of --dataset-csv)")
    p.add_argument("--prefix", default=None, help="Output file prefix (default: dataset file name without .csv[.gz])")
    p.add_argument("--subcellular-tsv", default=None,
                   help="HPA subcellular_location.tsv or .tsv.zip; downloaded from proteinatlas.org to this path "
                        "if it does not exist. Omit to skip the subcellular_* columns.")
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
    outputs = [("cell_annotations", cell_ann), ("image_annotations", img_ann), ("cell_types", vocab)]

    if args.subcellular_tsv:
        subcellular = load_subcellular(fetch_subcellular_location(args.subcellular_tsv))
        img_ann = attach_subcellular(img_ann, subcellular)
        outputs[1] = ("image_annotations", img_ann)
        outputs.append(("subcellular_locations", build_subcellular_vocabulary(img_ann)))

    n_without = int((img_ann["n_cell_types"] == 0).sum())
    logger.info("%d images, %d image x cell-type rows, %d cell types, %d images without any cell annotation",
                len(img_ann), len(cell_ann), len(vocab), n_without)
    for name, df in outputs:
        path = os.path.join(out_dir, f"{prefix}_{name}.csv")
        df.to_csv(path, index=False)
        logger.info("wrote %s (%d rows)", path, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
