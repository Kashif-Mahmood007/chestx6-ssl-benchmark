#!/usr/bin/env python3
"""
make_supp_tables_S2_S4.py -- Supplementary Tables S2 and S4 (Reviewer 3, Major Comments 3 and 7)

  S2  Absolute macro-F1 for every model under the clean condition and the eight synthetic image perturbations
      (mean +/- sample SD over the three seeds), plus the mean corrupted F1 and the worst-case F1, and the same two
      quantities expressed as performance RETAINED ABOVE CHANCE: (F1 - 1/6) / (clean F1 - 1/6) x 100, where 1/6 is
      the macro-F1 of random guessing among six classes of similar size (an approximation).
      Input: robustness_3seed_summary_full.csv   (the file you already have: two header rows, model key in col 1)

  S4  Source-domain and target-domain AUC-ROC with 95% intervals.
        * ChestX6 binary (normal vs any pathology) AUC with the bootstrap interval already computed in NB4/NB6
          (chestx6_binary_auc.csv);
        * ChestMNIST linear-probe and fine-tuned target AUC with the analytic Hanley-McNeil (1982) 95% interval,
          which reflects TEST-SET SAMPLING ONLY. Each target AUC is a single probe / fine-tuning run, so the interval
          does not include training randomness or the draw of the target training subset.
      Input: chestx6_binary_auc.csv, chestmnist_summary.csv

HOW TO RUN
    python make_supp_tables_S2_S4.py --robustness robustness_3seed_summary_full.csv \
        --source-auc chestx6_binary_auc.csv --target-auc chestmnist_summary.csv \
        --n-pos 10505 --n-neg 11928 --out /path/to/output_folder

  --n-pos / --n-neg are the numbers of pathology-positive and pathology-negative images in the ChestMNIST TEST split
  (22,433 images, 46.83% pathology -> about 10,505 and 11,928). Use the exact counts printed by NB6 if they differ.
  If the ViT-S/16 (Scratch) source AUC and interval are available, pass  --vit-source 0.99xx 0.99xx 0.99xx
  (AUC, lower, upper); otherwise that cell is written as [[ADD]] so that it cannot be overlooked.

OUTPUT   supp_S2_absolute_f1.tex  (label tab:S2_absolute_f1),  supp_S4_auc_intervals.tex  (label tab:S4_auc_ci),
         supp_S2_absolute_f1.csv, supp_S4_auc_intervals.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

KEYS = [("resnet50_scratch", "ResNet50 (Scratch)"), ("resnet50_pretrained", "ResNet50 (ImageNet)"),
        ("efficientnet_b0", "EfficientNet-B0"), ("mobilevit_xs", "MobileViT-XS"),
        ("vit_s16_scratch", "ViT-S/16 (Scratch)"), ("simclr_resnet50", "SimCLR + ResNet50"),
        ("mae_vits16", "MAE + ViT-S/16")]
CONDS = [("gaussian_noise_low", "Noise-L"), ("gaussian_noise_medium", "Noise-M"), ("brightness_up", "Bright+"),
         ("brightness_down", "Bright$-$"), ("gaussian_blur_mild", "Blur-M"), ("gaussian_blur_strong", "Blur-S"),
         ("contrast_reduction", "Contrast"), ("jpeg_artifacts", "JPEG")]


def hanley_mcneil_ci(auc, n_pos, n_neg, z=1.96):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = np.sqrt(var)
    return se, max(0.0, auc - z * se), min(1.0, auc + z * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", required=True)
    ap.add_argument("--source-auc", required=True)
    ap.add_argument("--target-auc", required=True)
    ap.add_argument("--n-pos", type=int, default=10505)
    ap.add_argument("--n-neg", type=int, default=11928)
    ap.add_argument("--vit-source", nargs=3, type=float, default=None, metavar=("AUC", "LO", "HI"))
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ------------------------------------------------------------------ S2
    r = pd.read_csv(a.robustness, header=[0, 1], index_col=0)
    rows, tex = [], []
    for key, label in KEYS:
        clean = (r.loc[key, ("clean", "mean")], r.loc[key, ("clean", "std")])
        cells = {c: (r.loc[key, (c, "mean")], r.loc[key, (c, "std")]) for c, _ in CONDS}
        means = [cells[c][0] for c, _ in CONDS]
        worst_i = int(np.argmin(means))
        chance = 1.0 / 6.0
        ret_mean = (float(np.mean(means)) - chance) / (clean[0] - chance) * 100
        ret_worst = (float(np.min(means)) - chance) / (clean[0] - chance) * 100
        rows.append(dict(model=label, clean_mean=clean[0], clean_sd=clean[1],
                         **{f"{c}_mean": cells[c][0] for c, _ in CONDS}, **{f"{c}_sd": cells[c][1] for c, _ in CONDS},
                         mean_corrupted_f1=float(np.mean(means)), worst_case_f1=float(np.min(means)),
                         worst_case_condition=CONDS[worst_i][1].replace("$-$", "-"),
                         retained_above_chance_mean_pct=ret_mean, retained_above_chance_worst_pct=ret_worst))
        tex.append(f"{label} & {clean[0]:.3f}$\\pm${clean[1]:.3f} & "
                   + " & ".join(f"{cells[c][0]:.3f}$\\pm${cells[c][1]:.3f}" for c, _ in CONDS)
                   + f" & {np.mean(means):.3f} & {np.min(means):.3f} ({CONDS[worst_i][1]}) & {ret_mean:.1f} & {ret_worst:.1f} \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "supp_S2_absolute_f1.csv"), index=False)
    head = (r"\textbf{Model} & \textbf{Clean} & " + " & ".join(f"\\textbf{{{n}}}" for _, n in CONDS)
            + r" & \textbf{Mean corrupted} & \textbf{Worst case} & \textbf{Retained, mean (\%)} & \textbf{Retained, worst (\%)} \\")
    S2 = "\n".join([r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                    r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * 13 + "}", r"\toprule", head, r"\midrule",
                    *tex, r"\bottomrule", r"\end{tabular}}",
                    r"\caption{\label{tab:S2_absolute_f1}Absolute test macro-F1 of the seven models (100\% labels) on the "
                    r"clean test set and under the eight synthetic image perturbations (mean $\pm$ sample SD over three "
                    r"fine-tuning seeds). Mean corrupted: unweighted mean of the eight condition means. Worst case: the "
                    r"lowest of the eight condition means (the condition is given in brackets). Retained: the mean corrupted and the worst-case F1 "
                    r"expressed as the percentage of the clean performance that lies above chance, $(F1-1/6)/(F1_{\text{clean}}-1/6)\times100$, "
                    r"with $1/6$ the macro-F1 of random guessing among six classes of similar size (an approximation). Noise-L/M: Gaussian noise "
                    r"$\sigma$\,=\,0.05/0.10; Bright+/$-$: brightness $\times$1.3/0.7; Blur-M/S: Gaussian blur "
                    r"$\sigma$\,=\,1.0/2.0; Contrast: factor 0.7; JPEG: quantisation approximating quality 50. The "
                    r"relative drops in the main text divide by the clean F1, which favours models with a low clean F1; "
                    r"the absolute values are given here so that the two views can be compared.}", r"\end{table}"]) + "\n"
    open(os.path.join(a.out, "supp_S2_absolute_f1.tex"), "w").write(S2)

    # ------------------------------------------------------------------ S4
    src = pd.read_csv(a.source_auc).set_index("Model")
    tgt = pd.read_csv(a.target_auc).set_index("Model")
    rows, tex = [], []
    for _, label in KEYS:
        if label in src.index:
            s = (src.loc[label, "ChestX6_binary_AUC"], src.loc[label, "CI_lo"], src.loc[label, "CI_hi"])
        elif label == "ViT-S/16 (Scratch)" and a.vit_source:
            s = tuple(a.vit_source)
        else:
            s = None
        lp, ft = float(tgt.loc[label, "LP AUC-ROC"]), float(tgt.loc[label, "FT AUC-ROC"])
        lp_se, lp_lo, lp_hi = hanley_mcneil_ci(lp, a.n_pos, a.n_neg)
        ft_se, ft_lo, ft_hi = hanley_mcneil_ci(ft, a.n_pos, a.n_neg)
        rows.append(dict(model=label, source_auc=None if s is None else s[0],
                         source_lo=None if s is None else s[1], source_hi=None if s is None else s[2],
                         lp_auc=lp, lp_se=lp_se, lp_lo=lp_lo, lp_hi=lp_hi, ft_auc=ft, ft_se=ft_se, ft_lo=ft_lo, ft_hi=ft_hi))
        sc = "[[ADD]]" if s is None else f"{s[0]:.4f} [{s[1]:.4f}, {s[2]:.4f}]"
        tex.append(f"{label} & {sc} & {lp:.3f} [{lp_lo:.3f}, {lp_hi:.3f}] & {ft:.3f} [{ft_lo:.3f}, {ft_hi:.3f}] \\\\")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "supp_S4_auc_intervals.csv"), index=False)
    S4 = "\n".join([r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
                    r"\textbf{Model} & \textbf{ChestX6 binary AUC} & \textbf{ChestMNIST, linear probe} & "
                    r"\textbf{ChestMNIST, fine-tuned} \\", r"\midrule", *tex, r"\bottomrule", r"\end{tabular}",
                    r"\caption{\label{tab:S4_auc_ci}AUC-ROC with 95\% intervals. ChestX6 binary: normal vs.\ any "
                    f"pathology, $P(\\text{{pathology}})=1-P(\\text{{normal}})$, percentile bootstrap over test images. "
                    f"ChestMNIST target AUC (test split, $n_{{\\text{{pos}}}}$\\,=\\,{a.n_pos:,}, "
                    f"$n_{{\\text{{neg}}}}$\\,=\\,{a.n_neg:,}): analytic Hanley--McNeil intervals, which reflect "
                    r"test-set sampling only. Each target AUC comes from a single linear-probe or fine-tuning run; the "
                    r"intervals do not include training randomness or the draw of the target training subset, and "
                    r"overlapping intervals must not be read as equivalence or non-equivalence of the models.}",
                    r"\end{table}"]).replace(f"{a.n_pos:,}", f"{a.n_pos:,}".replace(",", "{,}")) \
        .replace(f"{a.n_neg:,}", f"{a.n_neg:,}".replace(",", "{,}")) + "\n"
    open(os.path.join(a.out, "supp_S4_auc_intervals.tex"), "w").write(S4)

    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nWrote 4 files to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
