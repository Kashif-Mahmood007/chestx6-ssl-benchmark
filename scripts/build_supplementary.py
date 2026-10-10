#!/usr/bin/env python3
"""
build_supplementary.py -- builds the three files that go to the journal as Supplementary Information
=======================================================================================================

    Supplementary_Tables_S1-S5.pdf   Tables S1-S5 in one document        (journal: Supplementary Information)
    per_seed_results.csv             one row per training run (raw data of Table S1)
    effect_sizes.csv                 data of Table S3 (both SD conventions, exact p-values)
and, as an optional fourth upload, supplementary_scripts.zip (the code, because the paper says it is provided).

It runs the three table scripts (make_per_seed_results.py, make_supp_tables_S2_S4.py,
make_drs_uncertainty.py), writes the LaTeX wrapper, compiles the PDF twice and checks the result.
Nothing is estimated here; every number comes from the per-run JSON/CSV files of the results folder.

Usage (Google Colab or any machine with Python 3, pandas, scipy and pdflatex):

    python build_supplementary.py --results repo/results --scripts . --out supplementary_upload

The results folder must contain
    <model>_<budget>_seed<seed>.json  (63 runs of the seven models, plus the 18 train-only runs)
    robustness_3seed_summary_full.csv, chestx6_binary_auc.csv, chestmnist_summary.csv
The three table scripts must be in the --scripts folder.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

MAIN_KEYS = ["resnet50_scratch", "resnet50_pretrained", "efficientnet_b0", "mobilevit_xs",
             "vit_s16_scratch", "simclr_resnet50", "mae_vits16"]
TABLE_FILES = ["per_seed_macro_f1.tex", "supp_S2_absolute_f1.tex", "effect_sizes.tex",
               "supp_S4_auc_intervals.tex", "drs_uncertainty.tex"]          # = S1 ... S5, in this order
DELIVERABLES = ["Supplementary_Tables_S1-S5.pdf", "per_seed_results.csv", "effect_sizes.csv"]
SCRIPT_FILES = ["make_per_seed_results.py", "make_supp_tables_S2_S4.py", "make_drs_uncertainty.py",
                "build_supplementary.py", "count_tb_acquisitions.py", "check_mae_gradients.py",
                "NB3_MAE_pretrain_corrected_cell.py"]                      # optional 4th upload: the code

WRAPPER = r"""% Supplementary Tables S1-S5 (written by build_supplementary.py)
\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,amsmath,graphicx,pdflscape,caption}
\usepackage[T1]{fontenc}
\renewcommand{\thetable}{S\arabic{table}}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\newcommand{\inputtable}[2]{\IfFileExists{#1}{\input{#1}}{\par\textbf{[[Missing #1: run #2]]}\par}}
\begin{document}
\section*{Supplementary Tables}
\noindent All tables report the seven models of the main text. Test-set values; training seeds 42, 123 and 456.
Standard deviations use the sample form (divisor $n-1$) throughout.

\inputtable{per_seed_macro_f1.tex}{make_per_seed_results.py}      % S1
\begin{landscape}
\setlength{\textwidth}{\dimexpr\paperheight-4cm\relax}\setlength{\columnwidth}{\textwidth}\setlength{\linewidth}{\textwidth}
\inputtable{supp_S2_absolute_f1.tex}{make_supp_tables_S2_S4.py}   % S2 (wide: landscape page)
\end{landscape}
\inputtable{effect_sizes.tex}{make_per_seed_results.py}           % S3
\inputtable{supp_S4_auc_intervals.tex}{make_supp_tables_S2_S4.py} % S4
\inputtable{drs_uncertainty.tex}{make_drs_uncertainty.py}         % S5
\end{document}
"""


def run(cmd, cwd=None):
    print("  $", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:])
        print(res.stderr[-3000:])
        sys.exit(f"ERROR: command failed: {' '.join(cmd)}")
    return res.stdout


def need(path, what):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {what} not found: {path}")


def ensure_pdflatex():
    if shutil.which("pdflatex"):
        return
    if shutil.which("apt-get") and os.geteuid() == 0:
        print("pdflatex not found; installing LaTeX with apt-get (a few minutes, once) ...")
        subprocess.run("apt-get -qq update && apt-get -qq install -y texlive-latex-recommended "
                       "texlive-latex-extra texlive-fonts-recommended > /dev/null 2>&1", shell=True, check=False)
    if not shutil.which("pdflatex"):
        sys.exit("ERROR: pdflatex is not available. Install a LaTeX distribution (Colab: "
                 "!apt-get -qq install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="repo/results", help="folder with the per-run JSON and summary CSV files")
    ap.add_argument("--scripts", default=".", help="folder with the three make_*.py scripts")
    ap.add_argument("--out", default="supplementary_upload", help="folder that receives the three final files")
    ap.add_argument("--n-pos", type=int, default=10505, help="pathology-positive images in the ChestMNIST test split")
    ap.add_argument("--n-neg", type=int, default=11928, help="pathology-negative images in the ChestMNIST test split")
    a = ap.parse_args()

    results, scripts = os.path.abspath(a.results), os.path.abspath(a.scripts)
    out = os.path.abspath(a.out)
    work = os.path.join(out, "_build")
    os.makedirs(work, exist_ok=True)

    # ---- 0. inputs ---------------------------------------------------------------------------------
    for s in ("make_per_seed_results.py", "make_supp_tables_S2_S4.py", "make_drs_uncertainty.py"):
        need(os.path.join(scripts, s), "script " + s)
    robustness = os.path.join(results, "robustness_3seed_summary_full.csv")
    source_auc = os.path.join(results, "chestx6_binary_auc.csv")
    target_auc = os.path.join(results, "chestmnist_summary.csv")
    for p, w in ((robustness, "robustness summary"), (source_auc, "source AUC table"), (target_auc, "target AUC table")):
        need(p, w)
    ensure_pdflatex()

    # ---- 1. Tables S1, S3 and the two csv files -----------------------------------------------------
    print("\n[1/4] S1, S3, per_seed_results.csv, effect_sizes.csv")
    run([sys.executable, os.path.join(scripts, "make_per_seed_results.py"), results, "--out", work])
    runs = pd.read_csv(os.path.join(work, "per_seed_results.csv"))
    n_main = int(runs["model_key"].isin(MAIN_KEYS).sum())
    n_extra = len(runs) - n_main
    if n_main != 63:
        sys.exit(f"ERROR: {n_main} of the 63 runs of the seven models were found; the results folder is incomplete.")
    print(f"  OK: 63 runs of the seven models and {n_extra} train-only runs found")

    # ---- 2. Tables S2, S4 ----------------------------------------------------------------------------
    print("\n[2/4] S2, S4")
    run([sys.executable, os.path.join(scripts, "make_supp_tables_S2_S4.py"), "--robustness", robustness,
         "--source-auc", source_auc, "--target-auc", target_auc, "--n-pos", str(a.n_pos), "--n-neg", str(a.n_neg),
         "--out", work])

    # ---- 3. Table S5 ----------------------------------------------------------------------------------
    print("\n[3/4] S5")
    run([sys.executable, os.path.join(scripts, "make_drs_uncertainty.py"), "--out", work])

    # ---- 4. wrapper, PDF, checks ------------------------------------------------------------------------
    print("\n[4/4] PDF")
    for f in TABLE_FILES:
        need(os.path.join(work, f), "table file " + f)
        txt = open(os.path.join(work, f), encoding="utf-8").read()
        if "[[" in txt:
            sys.exit(f"ERROR: {f} still contains a [[...]] placeholder")
    open(os.path.join(work, "Supplementary_Tables_S1-S5.tex"), "w", encoding="utf-8").write(WRAPPER)
    log = ""
    for _ in range(2):                                   # twice: the table numbering settles on the second run
        res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                              "Supplementary_Tables_S1-S5.tex"], cwd=work, capture_output=True, text=True)
        log = res.stdout
        if res.returncode != 0:
            print(log[-3000:])
            sys.exit("ERROR: pdflatex failed (see above)")
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    pages = int(m.group(1)) if m else -1
    pdf = os.path.join(work, "Supplementary_Tables_S1-S5.pdf")
    need(pdf, "PDF")
    if "[[Missing" in log or "LaTeX Error" in log:
        sys.exit("ERROR: the LaTeX log reports a missing table or an error")
    print(f"  OK: PDF built, {pages} pages (expected 4), no placeholder, no LaTeX error")

    # ---- deliver the three files -------------------------------------------------------------------------
    for f in DELIVERABLES:
        shutil.copy(os.path.join(work, f), os.path.join(out, f))
    zpath = os.path.join(out, "supplementary_upload.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in DELIVERABLES:
            z.write(os.path.join(out, f), f)

    included = []
    with zipfile.ZipFile(os.path.join(out, "supplementary_scripts.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in SCRIPT_FILES:
            if os.path.exists(os.path.join(scripts, f)):
                z.write(os.path.join(scripts, f), "scripts/" + f)
                included.append(f)
    missing = [f for f in SCRIPT_FILES if f not in included]

    es = pd.read_csv(os.path.join(out, "effect_sizes.csv"))
    row = es[es["comparison"].str.startswith("Train-only")].iloc[0]
    print("\nCheck for the manuscript: train-only SimCLR vs ceiling, Cohen's d (population SD) = "
          f"{row['d_population_sd_n']:.3f}  (the paper says about 0.89)")
    print("\nFinal files in", out)
    for f in DELIVERABLES + ["supplementary_upload.zip", "supplementary_scripts.zip"]:
        print(f"  {f:34s} {os.path.getsize(os.path.join(out, f)):>9,} bytes")
    if missing:
        print("\nNot in supplementary_scripts.zip (upload them to Colab and run again if you want them): "
              + ", ".join(missing))
    print("\nUpload: the PDF as Supplementary Information; the two csv files and supplementary_scripts.zip as "
          "supplementary files; the two csv files also go to results/ on GitHub.")


if __name__ == "__main__":
    main()
