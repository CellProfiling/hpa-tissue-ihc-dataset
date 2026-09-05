#!/usr/bin/env python3
"""
Render the README figures.

  overview      one stained example image per tissue category, as a grid
  segmentation  original / tissue mask / crop for a few images that went through
                generate_tissue_masks.py and crop_images_to_masks.py

Example:
    python docs/make_figures.py overview --image-annotations HPA_full_dataset_idr_image_annotations.csv \\
        --root ./hpa_tissue --out docs/images/tissue_overview.jpg
    python docs/make_figures.py segmentation --root ./hpa_tissue --images 143063_A_6_4 152045_B_4_6 \\
        --out docs/images/segmentation_cropping.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_dataset_images as D  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def pick_examples(image_annotations, seed=0):
    """One image per tissue: the most strongly stained, then most annotated cell types; seeded tie-break."""
    df = pd.read_csv(image_annotations, dtype=str, keep_default_na=False)
    df["max_staining_code"] = df["max_staining_code"].astype(int)
    df["n_cell_types"] = df["n_cell_types"].astype(int)
    ranked = (df.sample(frac=1, random_state=seed)
                .sort_values(["tissue", "max_staining_code", "n_cell_types"], ascending=[True, False, False],
                             kind="mergesort"))
    return ranked.drop_duplicates("tissue")


def thumb(path, size):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    return im


def overview(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = pick_examples(args.image_annotations, args.seed)
    dataset = pd.read_csv(args.dataset_csv, dtype=str, keep_default_na=False).set_index("image_id")
    rows = dataset.loc[picks["image_id"]].reset_index().to_dict("records")
    session = requests.Session()
    for r in rows:
        res = D.download_row(r, args.root, session, ["idr", "hpa"])
        print(r["tissue"], r["image_id"], res["status"], res["message"], flush=True)
        if res["status"] == "failed":
            raise SystemExit(f"download failed for {r['image_id']}")

    n = len(rows)
    ncol = args.columns
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.75))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows):
        ax.imshow(thumb(D.dest_path(r, args.root), 300))
        ax.set_title(r["tissue"], fontsize=7, pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005, wspace=0.04, hspace=0.22)
    fig.savefig(args.out, dpi=150, pil_kwargs={"quality": 85, "optimize": True})
    print(f"wrote {args.out} ({n} tissues, {nrow}x{ncol})")


def segmentation(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(args.root)
    found = {p.stem: p for p in root.rglob("*.tif")}
    paths = [found[i] for i in args.images]
    fig, axes = plt.subplots(len(paths), 3, figsize=(3 * 3.0, len(paths) * 3.05))
    axes = np.atleast_2d(axes)
    for row, p in zip(axes, paths):
        img = np.asarray(Image.open(p).convert("RGB"))
        mask = np.load(p.parent / "masks" / f"{p.stem}_mask.npy")
        crop = np.load(p.parent / "crops" / f"{p.stem}_crop.npy")
        s = 8  # display downsampling
        row[0].imshow(img[::s, ::s]); row[0].set_title(f"{p.stem}  (3000x3000)", fontsize=8)
        row[1].imshow(img[::s, ::s]); row[1].contour(mask[::s, ::s], levels=[0.5], colors="red", linewidths=1.2)
        row[1].set_title(f"tissue mask  ({mask.mean() * 100:.0f} % of image)", fontsize=8)
        row[2].imshow(crop[::s, ::s]); row[2].set_title(f"crop  ({crop.shape[1]}x{crop.shape[0]}, padding 20)", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.18)
    fig.savefig(args.out, dpi=130, pil_kwargs={"quality": 85, "optimize": True})
    print(f"wrote {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overview")
    o.add_argument("--image-annotations", required=True, help="<dataset>_image_annotations.csv (build_annotations.py)")
    o.add_argument("--dataset-csv", required=True, help="the matching dataset CSV (for URLs and local_path)")
    o.add_argument("--root", required=True, help="download root")
    o.add_argument("--out", required=True)
    o.add_argument("--columns", type=int, default=9)
    o.add_argument("--seed", type=int, default=0)
    o.set_defaults(func=overview)
    s = sub.add_parser("segmentation")
    s.add_argument("--root", required=True, help="image root with masks/ and crops/ next to the TIFFs")
    s.add_argument("--images", nargs="+", required=True, help="image ids to show")
    s.add_argument("--out", required=True)
    s.set_defaults(func=segmentation)
    args = p.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
