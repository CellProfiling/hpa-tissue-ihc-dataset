#!/usr/bin/env python3
"""
Generate rough tissue masks for HPA IHC TIF images.

Input is either a dataset CSV (``--csv-file``, needs ``local_path`` and
``image_id`` columns; paths are ``<img-root>/<local_path>/<image_id>.tif``) or a
directory tree (``--root``, all ``*.tif`` below it). Masks are written next to
the images as ``<image_dir>/masks/<image_id>_mask.<png|pt|npy>``.

The script writes two report CSVs under ``<img-root>/metadata/`` (or ``<root>``):
``generate_tissue_masks_failed_<timestamp>.csv`` (images that raised) and
``generate_tissue_masks_no_tissue_<timestamp>.csv`` (images whose final mask is
empty). It exits non-zero when any image failed.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm


def fast_tissue_mask(
    pil_img,
    downsample=0.2,
    min_component_area=1000
):
    """
    Compute a very rough tissue mask outline with all internal cavities filled.

    Steps:
      1) Convert image to grayscale (max of RGB channels).
      2) Downsample for speed.
      3) Threshold with Otsu (invert so tissue=white), then morphological close/open.
      4) Keep only components above min_component_area. NOTE: the area is
         measured on the *downsampled* image, i.e. 1000 px at downsample=0.2
         corresponds to ~25 000 px at full resolution.
      5) Upsample back to original resolution.
      6) Fill any remaining holes (8-connected).
      7) Compute the convex hull of each tissue component for a rough outline.
      8) Perform a large closing to smooth the overall shape.
    """
    # Convert to grayscale
    img_np = np.array(pil_img)
    if img_np.ndim == 3:
        gray = img_np[..., :3].max(axis=-1)
    else:
        gray = img_np
    orig_h, orig_w = gray.shape

    # Downsample
    small = cv2.resize(
        gray, (0, 0), fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA
    )

    # Threshold and cleanup on small image
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    _, mask_small = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kern)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kern)

    # Filter small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_small, connectivity=8
    )
    clean_small = np.zeros_like(mask_small)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            clean_small[labels == i] = 255

    # Upsample
    clean_full = cv2.resize(
        clean_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    # Fill holes
    bin_full = (clean_full > 0).astype(np.uint8)
    filled = binary_fill_holes(bin_full, structure=np.ones((3, 3))).astype(np.uint8) * 255

    # Rough outline via convex hulls
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_mask = np.zeros_like(filled)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)

    # Smooth with a large closing
    big_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    rough_outline = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, big_kern)

    return rough_outline


def save_mask(mask, out_path, output_format):
    out_path = str(out_path)
    binary = (mask > 0).astype(np.uint8)
    if output_format == "png":
        Image.fromarray(binary * 255, mode="L").save(out_path)
    elif output_format == "pt":
        import torch  # only needed for this format
        torch.save(torch.from_numpy(binary), out_path)
    elif output_format == "npy":
        np.save(out_path, binary)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def get_mask_path(img_path, output_format):
    img_path = Path(img_path)
    mask_dir = img_path.parent / "masks"
    mask_dir.mkdir(exist_ok=True)
    suffix = {"pt": "_mask.pt", "npy": "_mask.npy", "png": "_mask.png"}[output_format]
    return mask_dir / f"{img_path.stem}{suffix}"


def process_image(args):
    """Return (img_path, status, message). status in {ok, skipped, no_tissue, failed}."""
    img_path, output_format, downsample, min_component_area = args
    try:
        out_path = get_mask_path(img_path, output_format)
        # A 0-byte file is a partial write from a killed job, not a finished mask.
        if out_path.exists() and out_path.stat().st_size > 0:
            return img_path, "skipped", "mask exists"
        with Image.open(img_path) as pil_img:
            mask = fast_tissue_mask(
                pil_img, downsample=downsample, min_component_area=min_component_area
            )
        save_mask(mask, out_path, output_format)
        if not mask.any():
            return img_path, "no_tissue", "empty mask"
        return img_path, "ok", ""
    except Exception as e:  # report, never abort the pool
        return img_path, "failed", f"{type(e).__name__}: {e}"


def get_image_paths_from_csv(csv_file, img_root=""):
    """
    Extract image paths from CSV file.

    Args:
        csv_file: Path to CSV file with columns 'local_path' and 'image_id'
        img_root: Root directory for images (optional)

    Returns:
        (existing_paths, missing_paths)
    """
    df = pd.read_csv(csv_file, dtype=str, keep_default_na=False)

    required_cols = ['local_path', 'image_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file missing required columns: {missing_cols}")

    existing, missing = [], []
    for local_path, image_id in zip(df['local_path'], df['image_id']):
        if os.path.isabs(local_path):
            img_path = os.path.join(local_path, f"{image_id}.tif")
        else:
            img_path = os.path.join(img_root, local_path, f"{image_id}.tif")
        (existing if os.path.exists(img_path) else missing).append(img_path)
    return existing, missing


def write_report(rows, path):
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "status", "message"])
        w.writerows(rows)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--root", type=str,
                             help="Root directory to recursively search for .tif files")
    input_group.add_argument("--csv-file", type=str,
                             help="CSV file with image paths (requires 'local_path' and 'image_id' columns)")
    parser.add_argument("--img-root", type=str, default=None,
                        help="Root directory for images when using --csv-file (required with --csv-file)")
    parser.add_argument("--output-format", choices=["png", "pt", "npy"], default="npy")
    parser.add_argument("--downsample", type=float, default=0.2)
    parser.add_argument("--min-component-area", type=int, default=1000,
                        help="Minimum connected-component area in DOWNSAMPLED pixels "
                             "(1000 at --downsample 0.2 is about 25 000 full-resolution pixels)")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--report-dir", type=str, default=None,
                        help="Where to write failed/no-tissue CSVs (default: <img-root>/metadata or <root>)")
    args = parser.parse_args(argv)

    missing = []
    if args.csv_file and not args.img_root:
        parser.error("--img-root is required with --csv-file")
    if args.csv_file:
        print(f"Reading image paths from CSV: {args.csv_file}")
        files, missing = get_image_paths_from_csv(args.csv_file, args.img_root)
        report_dir = args.report_dir or os.path.join(args.img_root, "metadata")
    else:
        print(f"Searching for .tif files in: {args.root}")
        files = [str(p) for p in Path(args.root).rglob("*.tif")]
        report_dir = args.report_dir or args.root
    for p in missing[:20]:
        print(f"Warning: image file not found: {p}")
    if len(missing) > 20:
        print(f"... {len(missing) - 20} more missing files")

    if args.max_images:
        files = files[: args.max_images]

    print(f"Processing {len(files)} images ({len(missing)} listed images missing on disk)")
    if len(files) == 0:
        print("No images found to process!")
        return 1 if missing else 0

    tasks = [
        (p, args.output_format, args.downsample, args.min_component_area)
        for p in files
    ]
    counts = {"ok": 0, "skipped": 0, "no_tissue": 0, "failed": 0}
    failed, no_tissue = [], []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for img_path, status, message in tqdm(ex.map(process_image, tasks, chunksize=8), total=len(tasks)):
            counts[status] += 1
            if status == "failed":
                failed.append((img_path, status, message))
            elif status == "no_tissue":
                no_tissue.append((img_path, status, message))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    failed_path = write_report(failed, os.path.join(report_dir, f"generate_tissue_masks_failed_{stamp}.csv"))
    no_tissue_path = write_report(no_tissue, os.path.join(report_dir, f"generate_tissue_masks_no_tissue_{stamp}.csv"))

    print(f"Done: ok={counts['ok']} skipped={counts['skipped']} "
          f"no_tissue={counts['no_tissue']} failed={counts['failed']} missing={len(missing)}")
    if failed_path:
        print(f"Failed images listed in: {failed_path}")
    if no_tissue_path:
        print(f"Empty-mask images listed in: {no_tissue_path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
