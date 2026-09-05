#!/usr/bin/env python3
"""
Download (or verify) the images listed in a dataset CSV produced by ``prepare_pilot_dataset.py``.

This is the collaborator-facing downloader: it needs only the dataset CSV
(columns ``image_id, local_path, idr_url, hpa_url`` and optionally ``set``) and
writes ``<root>/<local_path>/<image_id>.tif`` - the layout that
``generate_tissue_masks.py`` and ``crop_images_to_masks.py`` expect.

* Sources are tried in ``--source-order`` (default ``idr,hpa``). IDR answers
  ``200 image/tiff`` directly; HPA ``.tif`` links redirect (302) to the EBI
  BioStudies mirror, redirects are followed.
* Every file is written to a temp name, checked for the TIFF magic bytes and
  moved into place atomically. Existing valid TIFs are skipped, so re-running
  resumes an interrupted download.
* Failures are written to ``<root>/download_failed_<timestamp>.csv``; the exit
  code is 1 when anything failed.
* ``--verify`` downloads nothing: it opens and fully decodes every listed file
  and reports ``missing`` / ``corrupt`` files to ``<root>/verify_failed_<timestamp>.csv``.
  With ``--delete-corrupt`` undecodable files are removed so that a subsequent
  download run fetches them again.

Examples:
    python download_dataset_images.py --csv HPA_pilot_dataset_idr.csv --root ./hpa_tissue \\
        --set train val test --workers 8
    python download_dataset_images.py --csv HPA_pilot_dataset_idr.csv --root ./hpa_tissue --verify
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
URL_COLUMNS = {"idr": "idr_url", "hpa": "hpa_url"}


def dest_path(row, root):
    return os.path.join(root, row["local_path"], f"{row['image_id']}.tif")


def is_valid_tif(path):
    """Cheap check used while downloading: size and TIFF magic bytes only."""
    try:
        if os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            return f.read(4) in TIFF_MAGIC
    except OSError:
        return False


def decode_tif(path):
    """Full decode. Returns (ok, message) - catches truncated/corrupt files that pass the magic check."""
    if not is_valid_tif(path):
        return False, "missing TIFF header or file too small"
    try:
        with Image.open(path) as im:
            im.load()
            return True, f"{im.size[0]}x{im.size[1]} {im.mode}"
    except Exception as e:  # PIL raises many exception types for broken files
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def fetch(session, url, dest, timeout=60):
    """Stream url into dest via a temp file. Returns (ok, message)."""
    tmp = f"{dest}.{os.getpid()}.part"
    try:
        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            ctype = r.headers.get("content-type", "")
            if not ctype.startswith("image/"):
                return False, f"not an image ({ctype})"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if not is_valid_tif(tmp):
            os.remove(tmp)
            return False, "downloaded file is not a TIFF"
        os.replace(tmp, dest)
        return True, f"{os.path.getsize(dest)} bytes"
    except requests.RequestException as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, f"{type(e).__name__}: {e}"


def download_row(row, root, session, source_order, retries=2):
    """Return dict(image_id, status, source, message). status in {ok, skipped, failed}."""
    dest = dest_path(row, root)
    if is_valid_tif(dest):
        return {"image_id": row["image_id"], "status": "skipped", "source": "", "message": "exists"}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    attempts = []
    for source in source_order:
        url = row.get(URL_COLUMNS[source], "")
        if not url:
            continue
        for _ in range(retries + 1):
            ok, msg = fetch(session, url, dest)
            if ok:
                return {"image_id": row["image_id"], "status": "ok", "source": source, "message": msg}
            attempts.append(f"{source}: {msg}")
            if msg.startswith("HTTP 4"):
                break  # no point retrying a 404
    return {"image_id": row["image_id"], "status": "failed", "source": "",
            "message": "; ".join(attempts) or "no url"}


def verify_row(row, root, delete_corrupt=False):
    """Return dict(image_id, status, source, message). status in {ok, missing, corrupt}."""
    dest = dest_path(row, root)
    if not os.path.exists(dest):
        return {"image_id": row["image_id"], "status": "missing", "source": "", "message": dest}
    ok, msg = decode_tif(dest)
    if ok:
        return {"image_id": row["image_id"], "status": "ok", "source": "", "message": msg}
    if delete_corrupt:
        os.remove(dest)
        msg += " (deleted)"
    return {"image_id": row["image_id"], "status": "corrupt", "source": "", "message": msg}


def load_rows(csv_path, sets=None, max_images=None):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in ("image_id", "local_path") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV lacks columns {missing}")
    if not any(c in df.columns for c in URL_COLUMNS.values()):
        raise ValueError("CSV has neither idr_url nor hpa_url")
    if sets and "set" in df.columns:
        df = df[df["set"].isin(sets)]
    for col in URL_COLUMNS.values():
        if col not in df.columns:
            df[col] = ""
    if max_images:
        df = df.head(max_images)
    return df.to_dict("records")


def write_report(root, prefix, rows):
    path = os.path.join(root, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "status", "source", "message"])
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Dataset CSV from prepare_pilot_dataset.py")
    p.add_argument("--root", required=True, help="Destination root; files go to <root>/<local_path>/<image_id>.tif")
    p.add_argument("--source-order", default="idr,hpa", help="Comma-separated: idr, hpa (default idr,hpa)")
    p.add_argument("--set", nargs="*", default=None, help="Only these splits (train val test); default all")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--verify", action="store_true",
                   help="Do not download; fully decode every listed file and report missing/corrupt ones")
    p.add_argument("--delete-corrupt", action="store_true",
                   help="With --verify: delete files that fail to decode so the next download run re-fetches them")
    args = p.parse_args(argv)

    source_order = [s.strip() for s in args.source_order.split(",") if s.strip()]
    unknown = set(source_order) - set(URL_COLUMNS)
    if unknown:
        p.error(f"unknown source(s) {sorted(unknown)}; choose from {sorted(URL_COLUMNS)}")

    rows = load_rows(args.csv, args.set, args.max_images)
    os.makedirs(args.root, exist_ok=True)

    if args.verify:
        print(f"{len(rows)} images listed; verifying under {args.root}")
        bad_statuses, report_prefix = ("missing", "corrupt"), "verify_failed"

        def work(r):
            return verify_row(r, args.root, args.delete_corrupt)
    else:
        print(f"{len(rows)} images listed; destination {args.root}; sources {source_order}")
        session = requests.Session()
        session.headers.update({"User-Agent": "hpa-tissue-dataset-downloader/1.0"})
        bad_statuses, report_prefix = ("failed",), "download_failed"

        def work(r):
            return download_row(r, args.root, session, source_order, args.retries)

    counts, bad = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, r) for r in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            res = fut.result()
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            if res["status"] in bad_statuses:
                bad.append(res)

    report = write_report(args.root, report_prefix, bad) if bad else None
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"done: {summary}" + (f"  (problems: {report})" if report else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
