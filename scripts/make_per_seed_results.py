#!/usr/bin/env python3
"""
make_per_seed_results.py -- build the per-seed result files requested by Reviewer 3 (Major Comment 4).

WHAT IT DOES
  Reads the per-run JSON files written by notebooks NB2 / NB3 (one file per model x label budget x seed,
  e.g. resnet50_pretrained_100pct_seed42.json) and writes six files:

    per_seed_results.csv      one row per run: model, budget, seed, test macro-F1, accuracy, AUC, ECE, ...
    summary_by_model.csv      mean and BOTH standard deviations per model x budget
                              (sd_sample uses n-1, sd_population uses n)
    effect_sizes.csv          Cohen's d (pooled-SD form used in the paper) for the comparisons quoted in the
                              text, under both SD conventions, plus the paired standardised difference d_z
                              (for information only) and the EXACT two-sided p-values of the Wilcoxon
                              signed-rank test (seed-matched pairs) and of the rank-sum (Mann-Whitney) test
    effect_sizes.tex          the same comparisons as a LaTeX table (label tab:effect_sizes; "Supplementary
                              Table S3")
    per_seed_macro_f1.tex     a LaTeX table (7 models x 3 budgets x 3 seeds) for the supplement
                              (label tab:per_seed_f1; "Supplementary Table S1")
    table_cells_sample_sd.txt paste-ready "mean$\pm$SD" cells (sample SD, n-1) for the F1 columns of
                              Table 5 (3 decimals) and for the Primary / Train-Only columns of the train-only
                              validation table (4 decimals for train-only), so that every SD in the paper uses
                              the same convention (Tier 4, Item A0)

HOW TO RUN (Google Colab or any machine with Python 3 and pandas, numpy):
    python make_per_seed_results.py  /path/to/results  --out /path/to/output_folder

  /path/to/results is the folder that holds the *_seed*.json files (the "results/" folder of the repository).
  Files whose names contain "trainonly" or "tbcorrected" are read as separate experiments and are NOT mixed
  into the seven-model table.

WHY BOTH STANDARD DEVIATIONS
  NB2/NB3 print mean +/- std with the population form (divide by n = 3), while pandas' default (used for the
  robustness table) divides by n - 1. The two differ by a factor sqrt(3/2) = 1.22. Check the output and use ONE
  convention everywhere in the paper (see Tier 4, Item A0).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

MODELS = [  # (key used in file names, label used in the paper)
    ("resnet50_scratch", "ResNet50 (Scratch)"),
    ("resnet50_pretrained", "ResNet50 (ImageNet)"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("mobilevit_xs", "MobileViT-XS"),
    ("vit_s16_scratch", "ViT-S/16 (Scratch)"),
    ("simclr_resnet50", "SimCLR + ResNet50"),
    ("mae_vits16", "MAE + ViT-S/16"),
]
EXTRA = [("simclr_resnet50_trainonly", "SimCLR + ResNet50 (train-only SSL)"),
         ("mae_vits16_trainonly", "MAE + ViT-S/16 (train-only SSL)")]
BUDGETS = ["10pct", "20pct", "100pct"]
SEEDS = [42, 123, 456]


def load_runs(folder, keys):
    rows = []
    for key, label in keys:
        for b in BUDGETS:
            for s in SEEDS:
                path = os.path.join(folder, f"{key}_{b}_seed{s}.json")
                if not os.path.exists(path):
                    continue
                d = json.load(open(path))
                t = d.get("test_metrics", {})
                rows.append(dict(model_key=key, model=label, budget=b, seed=s,
                                 test_macro_f1=t.get("macro_f1"), test_accuracy=t.get("accuracy"),
                                 test_auc_ovr_macro=t.get("auc_ovr_macro"), test_ece=t.get("ece"),
                                 best_epoch=d.get("best_epoch"), best_val_macro_f1=d.get("best_val_f1")))
    return pd.DataFrame(rows)


def cohen_d(a, b, ddof):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.std(ddof=ddof) ** 2 + b.std(ddof=ddof) ** 2) / 2.0)
    return (a.mean() - b.mean()) / sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    main_df = load_runs(a.results_dir, MODELS)
    extra_df = load_runs(a.results_dir, EXTRA)
    if main_df.empty:
        sys.exit("No *_seed*.json files found for the seven models in " + a.results_dir)

    allruns = pd.concat([main_df, extra_df], ignore_index=True)
    allruns.to_csv(os.path.join(a.out, "per_seed_results.csv"), index=False)

    # ---- summary with both SD conventions
    g = allruns.groupby(["model", "budget"])["test_macro_f1"]
    summ = pd.DataFrame({"n_seeds": g.count(), "mean": g.mean(),
                         "sd_sample_n_minus_1": g.std(ddof=1), "sd_population_n": g.std(ddof=0)}).reset_index()
    summ.to_csv(os.path.join(a.out, "summary_by_model.csv"), index=False)

    missing = [(m, b, s) for m, _ in [(k, l) for k, l in MODELS] for b in BUDGETS for s in SEEDS
               if not ((main_df.model_key == m) & (main_df.budget == b) & (main_df.seed == s)).any()]
    if missing:
        print(f"WARNING: {len(missing)} of 63 runs for the seven models were not found, e.g. {missing[:3]}")

    # ---- effect sizes quoted in the paper (pooled-SD d, independent groups)
    def vals(key, b):
        return allruns[(allruns.model_key == key) & (allruns.budget == b)].sort_values("seed").test_macro_f1.values

    pairs = [("SimCLR vs ResNet50 (ImageNet), 100% labels", "simclr_resnet50", "resnet50_pretrained", "100pct"),
             ("SimCLR vs ResNet50 (ImageNet), 20% labels", "simclr_resnet50", "resnet50_pretrained", "20pct"),
             ("SimCLR vs ResNet50 (ImageNet), 10% labels", "simclr_resnet50", "resnet50_pretrained", "10pct"),
             ("SimCLR vs MAE, 10% labels", "simclr_resnet50", "mae_vits16", "10pct"),
             ("SimCLR vs MAE, 20% labels", "simclr_resnet50", "mae_vits16", "20pct"),
             ("SimCLR vs MAE, 100% labels", "simclr_resnet50", "mae_vits16", "100pct"),
             ("MAE vs ViT-S/16 (Scratch), 10% labels", "mae_vits16", "vit_s16_scratch", "10pct"),
             ("MAE vs ViT-S/16 (Scratch), 20% labels", "mae_vits16", "vit_s16_scratch", "20pct"),
             ("MAE vs ViT-S/16 (Scratch), 100% labels", "mae_vits16", "vit_s16_scratch", "100pct"),
             ("Train-only SimCLR vs ResNet50 (ImageNet), 100% labels",
              "simclr_resnet50_trainonly", "resnet50_pretrained", "100pct")]
    rows = []
    for name, ka, kb, b in pairs:
        va, vb = vals(ka, b), vals(kb, b)
        if len(va) == 3 and len(vb) == 3:
            diff = va - vb
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
            try:
                p_w = stats.wilcoxon(va, vb, method="exact", alternative="two-sided").pvalue
            except Exception:
                p_w = float("nan")
            try:
                p_u = stats.mannwhitneyu(va, vb, method="exact", alternative="two-sided").pvalue
            except Exception:
                p_u = float("nan")
            rows.append(dict(comparison=name, mean_diff=round(float(diff.mean()), 4),
                             d_sample_sd_n_minus_1=round(cohen_d(va, vb, 1), 3),
                             d_population_sd_n=round(cohen_d(va, vb, 0), 3),
                             d_paired_dz=round(float(dz), 3),
                             p_wilcoxon_exact=round(float(p_w), 3), p_ranksum_exact=round(float(p_u), 3)))
    es = pd.DataFrame(rows)
    es.to_csv(os.path.join(a.out, "effect_sizes.csv"), index=False)

    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"\textbf{Comparison} & \textbf{Mean diff.} & \textbf{Cohen's $d$} & \textbf{Wilcoxon $p$} & \textbf{Rank-sum $p$} \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['comparison'].replace('%', chr(92) + '%')} & {r['mean_diff']:+.4f} & {r['d_sample_sd_n_minus_1']:.2f} & "
                     f"{r['p_wilcoxon_exact']:.2f} & {r['p_ranksum_exact']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\label{tab:effect_sizes}Mean difference in test macro-F1, Cohen's $d$ (pooled sample SD, "
              r"independent groups, three training seeds per model) and exact two-sided $p$-values for the "
              r"comparisons quoted in the text. With three seeds per model the smallest attainable exact $p$ is "
              r"0.25 (signed-rank on seed-matched pairs) or 0.10 (rank-sum), so neither test can reach $0.05$; "
              r"the values are descriptive. The paired standardised difference $d_z$ is in "
              r"\texttt{effect\_sizes.csv}.}", r"\end{table}"]
    open(os.path.join(a.out, "effect_sizes.tex"), "w").write("\n".join(lines) + "\n")

    # ---- LaTeX per-seed table
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
             r"\textbf{Model} & \multicolumn{3}{c}{\textbf{10\% labels}} & \multicolumn{3}{c}{\textbf{20\% labels}} "
             r"& \multicolumn{3}{c}{\textbf{100\% labels}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r" & s42 & s123 & s456 & s42 & s123 & s456 & s42 & s123 & s456 \\", r"\midrule"]
    for key, label in MODELS:
        cells = []
        for b in BUDGETS:
            for s in SEEDS:
                r = main_df[(main_df.model_key == key) & (main_df.budget == b) & (main_df.seed == s)]
                cells.append(f"{r.test_macro_f1.iloc[0]:.4f}" if len(r) else "--")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\label{tab:per_seed_f1}Test-set macro-F1 for every training run (seven models, three label "
              r"budgets, seeds 42, 123 and 456). SimCLR and MAE encoders were pretrained once; only fine-tuning was "
              r"repeated across seeds. The same values, together with accuracy, AUC and ECE, are in "
              r"\texttt{per\_seed\_results.csv} in the code archive.}", r"\end{table}"]
    open(os.path.join(a.out, "per_seed_macro_f1.tex"), "w").write("\n".join(lines) + "\n")

    # ---- paste-ready cells (sample SD, n-1) for Table 5 and the train-only table
    def cell(model, b, nd):
        r = summ[(summ.model == model) & (summ.budget == b)]
        if r.empty:
            return "--"
        return f"{r['mean'].iloc[0]:.{nd}f}$\\pm${r['sd_sample_n_minus_1'].iloc[0]:.{nd}f}"

    out_lines = ["% Table 5 (tab:main_results): F1@10% | F1@20% | F1@100%   (mean$\\pm$sample SD, 3 decimals)"]
    for _, label in MODELS:
        out_lines.append(f"{label:22s} " + " | ".join(cell(label, b, 3) for b in BUDGETS))
    out_lines += ["", "% Train-only validation table (tab:trainonly_validation): Primary F1 (3 dec.) | Train-Only F1 (4 dec.)"]
    for base, tag in [("SimCLR + ResNet50", "SimCLR + ResNet50 (train-only SSL)"),
                      ("MAE + ViT-S/16", "MAE + ViT-S/16 (train-only SSL)")]:
        for b in BUDGETS:
            out_lines.append(f"{base:20s} {b:7s} " + cell(base, b, 3) + " | " + cell(tag, b, 4))
    open(os.path.join(a.out, "table_cells_sample_sd.txt"), "w").write("\n".join(out_lines) + "\n")

    pd.set_option("display.width", 200)
    print("\n== Mean and SD of test macro-F1 (both conventions) ==")
    print(summ.round(4).to_string(index=False))
    print("\n== Cohen's d (pooled-SD, independent groups) ==")
    print(es.to_string(index=False))
    print(f"\nWrote 6 files to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
