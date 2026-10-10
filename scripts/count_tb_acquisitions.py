#!/usr/bin/env python3
"""
Count original Tuberculosis (TB) acquisitions per split  (Reviewer 3, Comment 1c)
=================================================================================

Input: ONE csv file with one row per Tuberculosis image (3,173 rows) and these columns
(the default column names can be changed with the options below):

    filename        image file name                              (needed only if acq_id is not a column)
    acq_id          original acquisition the image comes from    (or let --acq-regex extract it from filename)
    split_original  train / val / test in the split used for all main experiments
    split_grouped   train / val / test in the group-respecting split used for the retrains

Usage:
    python count_tb_acquisitions.py tb_manifest.csv
    python count_tb_acquisitions.py tb_manifest.csv --acq-regex "Tuberculosis-(\\d+)"   # acq_id taken from filename

The script prints the six numbers for the manuscript and the response letter, and it CHECKS them
against the totals already stated in the manuscript (3,173 images, 700 acquisitions, 545 acquisitions
in more than one partition, 2,545 affected images, 635 test images). If any check fails the numbers
must NOT be used: the manifest is not the one behind the published audit.
"""
import argparse
import re
import sys

import pandas as pd

SPLITS = ("train", "val", "test")
ALIASES = {"train": "train", "training": "train",
           "val": "val", "valid": "val", "validation": "val",
           "test": "test", "testing": "test"}

# totals already stated in the manuscript (Section 5.3, Limitation 2)
EXPECTED = {"TB images": 3173, "original acquisitions": 700,
            "acquisitions in >1 partition": 545, "images of those acquisitions": 2545,
            "test TB images": 635}


def clean_split(series, name):
    lab = series.astype(str).str.strip().str.lower().map(ALIASES)
    bad = series[lab.isna()].astype(str).unique()
    if len(bad):
        sys.exit(f"ERROR: column '{name}' has labels other than train/val/test: {list(bad)[:10]}")
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="TB manifest (one row per Tuberculosis image)")
    ap.add_argument("--filename", default="filename")
    ap.add_argument("--acq", default="acq_id")
    ap.add_argument("--orig", default="split_original")
    ap.add_argument("--grouped", default="split_grouped")
    ap.add_argument("--acq-regex", default=None,
                    help="regex with ONE group that extracts the acquisition id from the file name "
                         "(used only when the acq column is absent)")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    print(f"read {len(df)} rows; columns: {list(df.columns)}")

    # ---- acquisition id -------------------------------------------------------------------
    if a.acq in df.columns:
        acq = df[a.acq].astype(str)
    elif a.acq_regex:
        if a.filename not in df.columns:
            sys.exit(f"ERROR: no '{a.acq}' and no '{a.filename}' column to apply --acq-regex to")
        ext = df[a.filename].astype(str).str.extract(a.acq_regex, expand=False)
        if ext.isna().any():
            sys.exit(f"ERROR: --acq-regex did not match {int(ext.isna().sum())} file names, e.g. "
                     f"{df.loc[ext.isna(), a.filename].head(5).tolist()}")
        acq = ext.astype(str)
    else:
        sys.exit(f"ERROR: the file has no '{a.acq}' column; give --acq-regex to extract it from '{a.filename}'")

    for c in (a.orig, a.grouped):
        if c not in df.columns:
            sys.exit(f"ERROR: column '{c}' not found; available: {list(df.columns)} (use --orig / --grouped)")
    d = pd.DataFrame({"acq": acq, "orig": clean_split(df[a.orig], a.orig),
                      "grp": clean_split(df[a.grouped], a.grouped)})

    # ---- counts -----------------------------------------------------------------------------
    N = {s: int(d.loc[d.orig == s, "acq"].nunique()) for s in SPLITS}      # original split
    G = {s: int(d.loc[d.grp == s, "acq"].nunique()) for s in SPLITS}       # group-respecting split
    parts = d.groupby("acq")["orig"].nunique()                              # partitions per acquisition
    spanning = parts[parts > 1].index
    found = {"TB images": len(d), "original acquisitions": int(d["acq"].nunique()),
             "acquisitions in >1 partition": int(len(spanning)),
             "images of those acquisitions": int(d["acq"].isin(spanning).sum()),
             "test TB images": int((d.orig == "test").sum())}

    problems = []
    # internal consistency (must hold for any correct manifest)
    if d.groupby("acq")["grp"].nunique().max() != 1:
        problems.append("group-respecting split puts some acquisition in more than one partition")
    if sum(G.values()) != found["original acquisitions"]:
        problems.append("grouped counts do not sum to the number of acquisitions")
    if sum(N.values()) != int(parts.sum()):
        problems.append("original-split counts do not equal the sum of partitions per acquisition")

    print("\nCheck against the totals already printed in the manuscript:")
    for k, want in EXPECTED.items():
        ok = found[k] == want
        print(f"  {'OK      ' if ok else 'MISMATCH'}  {k:32s} found {found[k]:>6,}   manuscript {want:>6,}")
        if not ok:
            problems.append(f"{k}: found {found[k]:,}, manuscript says {want:,}")
    pct = 100 * found["images of those acquisitions"] / found["TB images"]
    print(f"            affected share                  {pct:.2f}%  (manuscript 80.21%)")
    if round(pct, 2) != 80.21:
        problems.append(f"affected share {pct:.2f}% differs from 80.21%")

    print("\nTB images per partition   original split:",
          {s: int((d.orig == s).sum()) for s in SPLITS},
          "  grouped split:", {s: int((d.grp == s).sum()) for s in SPLITS})
    print(f"Original-split sum {sum(N.values()):,} (must lie between "
          f"{found['original acquisitions'] + found['acquisitions in >1 partition']:,} and "
          f"{found['original acquisitions'] + 2 * found['acquisitions in >1 partition']:,})")

    if problems:
        print("\n*** DO NOT USE THESE NUMBERS ***")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    print("\nALL CHECKS PASSED. Fill the markers with:")
    print(f"  N_ACQ_TRAIN = {N['train']:,}   N_ACQ_VAL = {N['val']:,}   N_ACQ_TEST = {N['test']:,}   (original split)")
    print(f"  G_ACQ_TRAIN = {G['train']:,}   G_ACQ_VAL = {G['val']:,}   G_ACQ_TEST = {G['test']:,}   (group-respecting split)")
    print("\nLaTeX text for Section 5.3, Limitation 2:")
    print(f"  the training, validation and test partitions of the original split contain {N['train']:,}, "
          f"{N['val']:,} and {N['test']:,} distinct original acquisitions, respectively (these sum to more than "
          f"700 because 545 acquisitions occur in more than one partition), and the group-respecting split "
          f"contains {G['train']:,}, {G['val']:,} and {G['test']:,} (summing to 700)")


if __name__ == "__main__":
    main()
