#!/usr/bin/env python3
"""
Release checks for the train/val/test split of dataset CSVs from prepare_pilot_dataset.py.

Run on the real CSVs after every regeneration; exits 1 if a check fails. Checks per dataset:
  * every image has a set, all three sets are present, no antibody or gene is in two sets;
  * per tissue, the image share of each set is within --tissue-tol percentage points of the target fractions;
  * per set, the share of antibodies in each source x antibody-number stratum (CAB/HPA x quintile, the proxy for
    staining era used by the split) and in each source is within --strata-tol points of its share over all antibodies;
  * optional (--metadata NAME=CSV with image_id and stained): the staining-year distribution of val and test
    differs from train by at most --year-tvd total-variation distance (only aggregates are printed).
Across datasets: a gene that occurs in several datasets has the same set in all of them.

Example:
    python check_dataset_splits.py --dataset pilot=HPA_pilot_dataset.csv --dataset full=HPA_full_dataset.csv
"""

import argparse
import itertools
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_pilot_dataset import SET_NAMES, antibody_strata, read_csv  # noqa: E402

logger = logging.getLogger("check_dataset_splits")


def _named(values, what):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"--{what} expects NAME=PATH, got {v!r}")
        name, path = v.split("=", 1)
        out[name] = path
    return out


def check_dataset(name, df, fracs, tissue_tol, strata_tol, stained=None, year_tvd=0.10):
    """Return a list of (check, ok, detail) for one dataset frame (image_id, antibody_id, ensembl_id, tissue, set)."""
    res = []
    target = dict(zip(SET_NAMES, fracs))
    bad_set = ~df["set"].isin(SET_NAMES)
    res.append(("every image has a set", not bad_set.any(), f"{int(bad_set.sum())} rows without a valid set"))
    present = set(df["set"]) & set(SET_NAMES)
    res.append(("all sets present", present == set(SET_NAMES), f"present: {sorted(present)}"))
    for key in ("antibody_id", "ensembl_id"):
        n = int((df.groupby(key)["set"].nunique() > 1).sum())
        res.append((f"no {key} in two sets", n == 0, f"{n} span several sets"))

    share = pd.crosstab(df["tissue"], df["set"], normalize="index").reindex(columns=list(SET_NAMES), fill_value=0)
    dev = max((share[k] - v).abs().max() for k, v in target.items()) * 100
    worst = max(((t, k, (share.loc[t, k] - target[k]) * 100) for t in share.index for k in SET_NAMES),
                key=lambda x: abs(x[2]))
    res.append((f"tissue share within {tissue_tol} pt", dev <= tissue_tol,
                f"max |dev| {dev:.2f} pt ({worst[0]}, {worst[1]} {worst[2]:+.2f})"))

    ab = df.drop_duplicates("antibody_id")[["antibody_id", "set"]].copy()
    ab["stratum"] = antibody_strata(ab["antibody_id"])
    overall = ab["stratum"].value_counts(normalize=True)
    per_set = pd.crosstab(ab["stratum"], ab["set"], normalize="columns").reindex(columns=list(SET_NAMES), fill_value=0)
    sdev = (per_set.sub(overall, axis=0)).abs().max().max() * 100
    src = ab["antibody_id"].str.extract(r"^([A-Za-z]+)")[0].fillna("other")
    src_share = pd.crosstab(src, ab["set"], normalize="columns").reindex(columns=list(SET_NAMES), fill_value=0)
    src_dev = (src_share.sub(src.value_counts(normalize=True), axis=0)).abs().max().max() * 100
    src_txt = "; ".join(f"{s} " + "/".join(f"{100 * src_share.loc[s, k]:.1f}" for k in SET_NAMES)
                        for s in src_share.index)
    res.append((f"source x number strata within {strata_tol} pt", sdev <= strata_tol and src_dev <= strata_tol,
                f"max |dev| strata {sdev:.2f} pt, source {src_dev:.2f} pt; source share train/val/test: {src_txt}"))

    if stained is not None:
        year = df["image_id"].map(stained).str[:4]
        miss = int((year.isna() | (year == "")).sum())
        dist = pd.crosstab(year, df["set"], normalize="columns").reindex(columns=list(SET_NAMES), fill_value=0)
        tvd = {k: 0.5 * (dist[k] - dist["train"]).abs().sum() for k in ("val", "test")}
        y0910 = {k: 100 * year[df["set"] == k].isin(["2009", "2010"]).mean() for k in SET_NAMES}
        res.append((f"staining-year TVD to train <= {year_tvd}", max(tvd.values()) <= year_tvd,
                    f"TVD val {tvd['val']:.3f} test {tvd['test']:.3f}; stained 2009-10 "
                    + "/".join(f"{y0910[k]:.1f}" for k in SET_NAMES) + f" %; {miss} images without date"))
    return res


def check_gene_agreement(frames):
    res = []
    for (a, da), (b, db) in itertools.combinations(frames.items(), 2):
        ga = da.drop_duplicates("ensembl_id").set_index("ensembl_id")["set"]
        gb = db.drop_duplicates("ensembl_id").set_index("ensembl_id")["set"]
        common = ga.index.intersection(gb.index)
        n = int((ga[common] != gb[common]).sum())
        res.append((f"{a} vs {b}: same set per gene", n == 0, f"{n} of {len(common)} shared genes differ"))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, help="NAME=CSV (repeatable)")
    p.add_argument("--metadata", action="append", help="NAME=CSV with image_id and stained (repeatable, optional)")
    p.add_argument("--fracs", type=float, nargs=3, default=(0.70, 0.10, 0.20), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--tissue-tol", type=float, default=1.0, help="percentage points (default 1.0)")
    p.add_argument("--strata-tol", type=float, default=1.5, help="percentage points (default 1.5)")
    p.add_argument("--year-tvd", type=float, default=0.10, help="max total-variation distance (default 0.10)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    datasets, metadata = _named(args.dataset, "dataset"), _named(args.metadata, "metadata")
    cols = ["image_id", "antibody_id", "ensembl_id", "tissue", "set"]
    frames = {n: read_csv(path)[cols] for n, path in datasets.items()}
    results = []
    for n, df in frames.items():
        stained = None
        if n in metadata:
            m = read_csv(metadata[n])[["image_id", "stained"]].drop_duplicates("image_id")
            stained = m.set_index("image_id")["stained"]
        results += [(n,) + r for r in check_dataset(n, df, args.fracs, args.tissue_tol, args.strata_tol,
                                                    stained, args.year_tvd)]
    if len(frames) > 1:
        results += [("cross",) + r for r in check_gene_agreement(frames)]
    for name, check, ok, detail in results:
        logger.info("%s %-6s %-40s %s", "PASS" if ok else "FAIL", name, check, detail)
    failed = sum(not ok for _, _, ok, _ in results)
    logger.info("%d checks, %d failed", len(results), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
