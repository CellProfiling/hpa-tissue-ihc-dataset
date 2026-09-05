#!/usr/bin/env python3
"""
Build the IDR normal-tissue image table with FTP paths.

Regenerates ``data/external/idr0043-experimentA-annotation-M-00100_ftp_paths.csv``
from inputs that are checked into / kept next to the repo:

1. The IDR idr0043 annotation CSVs (``idr0043-experimentA-annotation-run*.csv``),
   one per HPA release batch. Their column sets differ slightly between runs;
   only the 22 columns shared by all runs are kept (see ``OUTPUT_COLUMNS``).
2. A recursive listing of the IDR FTP tree
   (``ftp.ebi.ac.uk/pub/databases/IDR/idr0043-uhlen-humanproteinatlas/``),
   one absolute path per line.

Rows are kept when the SNOMED pathology accession contains ``M-00100``
("Normal tissue, NOS"). Each row gets ``source_file`` (which run CSV it came
from) and ``ftp_path`` (``<batch>/<antibody>/<image>.tif`` relative to the
idr0043 root). When an image appears in several batches the *last* listing
entry wins (the listing is sorted, so this is the newest batch); this matches
the historical table.

Usage:
    python build_idr_ftp_paths.py \
        --idr-dir data/external/IDR \
        --listing data/external/IDR_all_paths_raw.txt \
        --output data/external/idr0043-experimentA-annotation-M-00100_ftp_paths.csv
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)

IDR_ROOT_SEGMENT = "idr0043-uhlen-humanproteinatlas/"
NORMAL_TISSUE_ACCESSION = "M-00100"
PATHOLOGY_COLUMN = "Characteristics [Pathology] Accession"
IMAGE_NAME_COLUMN = "Image Name"

# Columns shared by all annotation runs, in the historical output order.
OUTPUT_COLUMNS = [
    "Dataset Name",
    "Image Name",
    "Term Source REF",
    "Characteristics [Organism Part]",
    "Characteristics [Organism Part] Accession",
    "Characteristics [Pathology]",
    "Characteristics [Pathology] Accession",
    "Characteristics [Sex]",
    "Characteristics [Age]",
    "Characteristics [Individual]",
    "Assay Name",
    "Antibody identifier",
    "Antibody dilution",
    "Retrieval method",
    "Human Protein Atlas version",
    "Source Name",
    "Characteristics [Organism]",
    "Comment [Image File Path]",
    "Comment [Gene Identifier] 1",
    "Comment [Gene Identifier] 2",
    "Comment [Gene Symbol] 1",
    "Comment [Gene Symbol] 2",
]


def load_ftp_listing(listing_path):
    """Map image basename -> ftp path relative to the idr0043 root.

    Only ``.tif`` entries are used. Later lines overwrite earlier ones, so for
    images present in several batches the last (newest) batch is kept.
    """
    mapping = {}
    n_lines = 0
    with open(listing_path, "r") as f:
        for line in f:
            line = line.strip()
            n_lines += 1
            if not line.lower().endswith(".tif"):
                continue
            idx = line.find(IDR_ROOT_SEGMENT)
            rel = line[idx + len(IDR_ROOT_SEGMENT):] if idx >= 0 else line.lstrip("/")
            mapping[os.path.basename(rel)] = rel
    logger.info("Listing: %d lines, %d unique TIF basenames", n_lines, len(mapping))
    return mapping


def load_normal_tissue_rows(csv_path):
    """Read one annotation run and keep normal-tissue rows with the shared columns."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, low_memory=False)
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks expected columns: {missing}")
    keep = df[PATHOLOGY_COLUMN].str.contains(NORMAL_TISSUE_ACCESSION, regex=False)
    out = df.loc[keep, OUTPUT_COLUMNS].copy()
    out["source_file"] = os.path.basename(csv_path)
    logger.info("%s: %d rows, %d normal-tissue", os.path.basename(csv_path), len(df), len(out))
    return out


def build_table(idr_dir, listing_path):
    run_files = sorted(glob.glob(os.path.join(idr_dir, "idr0043-experimentA-annotation-run*.csv")))
    if not run_files:
        raise FileNotFoundError(f"No annotation run CSVs under {idr_dir}")
    frames = [load_normal_tissue_rows(p) for p in run_files]
    table = pd.concat(frames, ignore_index=True)

    ftp_map = load_ftp_listing(listing_path)
    table["ftp_path"] = table[IMAGE_NAME_COLUMN].map(ftp_map).fillna("")
    n_missing = int((table["ftp_path"] == "").sum())
    logger.info("Table: %d rows, %d without an FTP path", len(table), n_missing)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parser.add_argument("--idr-dir", default=os.path.join(repo_root, "data", "external", "IDR"),
                        help="Directory with idr0043-experimentA-annotation-run*.csv")
    parser.add_argument("--listing", default=os.path.join(repo_root, "data", "external", "IDR_all_paths_raw.txt"),
                        help="Recursive FTP listing of the idr0043 tree (one path per line)")
    parser.add_argument("--output",
                        default=os.path.join(repo_root, "data", "external",
                                             "idr0043-experimentA-annotation-M-00100_ftp_paths.csv"),
                        help="Output CSV path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table = build_table(args.idr_dir, args.listing)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
