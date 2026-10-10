#!/usr/bin/env python3
"""
make_drs_uncertainty.py -- seed-level uncertainty of the Deployment Risk Score (Reviewer 3, Major Comment 5)

WHAT IT DOES
  For every model and each of the three DRS scenarios it computes
     DRS = w_acc * F1@budget * 100 + w_rob * (100 - AvgDrop)
  and its uncertainty from the three training seeds:
     * SD of a single-run DRS (first-order propagation of the F1 SD and the Avg-Drop SD, assumed independent)
     * 95% interval = DRS +/- 4.303 * SD / sqrt(3)   (t distribution, 2 degrees of freedom)
     * P(rank 1)  = share of 20,000 Monte-Carlo draws in which the model has the highest DRS; each model's
                    true-mean DRS is drawn as  DRS + SEM_F1 * t2  -  SEM_drop * t2  (t2 = Student-t, 2 d.f.),
                    which is deliberately heavy-tailed (conservative).
  Only TRAINING-SEED variation is represented. Test-set sampling, the labelled-subset draw, the weights and
  pretraining-run variability (the SimCLR and MAE encoders were pretrained once) are NOT included.

  Writes   drs_uncertainty.csv   and   drs_uncertainty.tex   (LaTeX table, label tab:drs_uncertainty,
  "Supplementary Table S5").

HOW TO RUN
  1. With the values printed in the paper (no extra files needed; this reproduces the numbers in the paper):
         python make_drs_uncertainty.py --out /path/to/output_folder
     The built-in F1 SDs are the Table-5 SDs (population form, divided by n); they are converted to the sample
     form (n - 1) by multiplying by sqrt(3/2). The Avg-Drop SDs are already sample SDs (Table 7).
  2. From your own per-seed files (preferred once you have run make_per_seed_results.py):
         python make_drs_uncertainty.py --summary summary_by_model.csv \
                --robustness robustness_3seed_summary_full.csv --out /path/to/output_folder
     The F1 mean and SD (n - 1) are then read from summary_by_model.csv and the Avg-Drop mean and SD from the
     robustness summary CSV (two header rows, model key in the first column).

  Requires numpy and pandas. The random seed is fixed so that the table is reproducible.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

# ---- built-in values (Table 5 means and population SDs; Table 7 Avg-Drop mean and sample SD)
F1_BUILTIN = {
    "ResNet50 (Scratch)":  {"10": (0.767, 0.006),   "20": (0.771, 0.065),  "100": (0.898, 0.003)},
    "ResNet50 (ImageNet)": {"10": (0.848, 0.009),   "20": (0.878, 0.010),  "100": (0.950, 0.012)},
    "EfficientNet-B0":     {"10": (0.830, 0.003),   "20": (0.872, 0.005),  "100": (0.948, 0.005)},
    "MobileViT-XS":        {"10": (0.835, 0.004),   "20": (0.860, 0.008),  "100": (0.941, 0.004)},
    "ViT-S/16 (Scratch)":  {"10": (0.6721, 0.0002), "20": (0.7211, 0.0120), "100": (0.8144, 0.0041)},
    "SimCLR + ResNet50":   {"10": (0.843, 0.005),   "20": (0.867, 0.005),  "100": (0.952, 0.002)},
    "MAE + ViT-S/16":      {"10": (0.607, 0.044),   "20": (0.676, 0.022),  "100": (0.833, 0.015)},
}
DROP_BUILTIN = {"ResNet50 (Scratch)": (4.25, 1.16), "ResNet50 (ImageNet)": (16.66, 2.25),
                "EfficientNet-B0": (39.97, 2.59), "MobileViT-XS": (27.54, 3.15),
                "ViT-S/16 (Scratch)": (2.29, 0.16), "SimCLR + ResNet50": (13.14, 0.30),
                "MAE + ViT-S/16": (2.31, 0.24)}
KEYS = {"resnet50_scratch": "ResNet50 (Scratch)", "resnet50_pretrained": "ResNet50 (ImageNet)",
        "efficientnet_b0": "EfficientNet-B0", "mobilevit_xs": "MobileViT-XS",
        "vit_s16_scratch": "ViT-S/16 (Scratch)", "simclr_resnet50": "SimCLR + ResNet50",
        "mae_vits16": "MAE + ViT-S/16"}
SCENARIOS = {"A": ("100", 0.70, 0.30), "B": ("20", 0.50, 0.50), "C": ("10", 0.30, 0.70)}
SCEN_NAME = {"A": "A (100\\% labels, $w_{\\mathrm{acc}}$=0.70)", "B": "B (20\\% labels, $w_{\\mathrm{acc}}$=0.50)",
             "C": "C (10\\% labels, $w_{\\mathrm{acc}}$=0.30)"}


def load_inputs(summary, robustness):
    """Return F1[model][budget] = (mean, sample SD) and DROP[model] = (mean, sample SD)."""
    k = np.sqrt(1.5)
    if summary is None:
        F1 = {m: {b: (v[0], v[1] * k) for b, v in d.items()} for m, d in F1_BUILTIN.items()}
    else:
        s = pd.read_csv(summary)
        F1 = {}
        for m in KEYS.values():
            F1[m] = {}
            for b in ("10", "20", "100"):
                r = s[(s.model == m) & (s.budget == f"{b}pct")]
                F1[m][b] = (float(r["mean"].iloc[0]), float(r["sd_sample_n_minus_1"].iloc[0]))
    if robustness is None:
        DROP = dict(DROP_BUILTIN)
    else:
        r = pd.read_csv(robustness, header=[0, 1], index_col=0)
        DROP = {KEYS[key]: (float(r.loc[key, ("avg_drop_pct", "mean")]), float(r.loc[key, ("avg_drop_pct", "std")]))
                for key in KEYS}
    return F1, DROP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None, help="summary_by_model.csv from make_per_seed_results.py")
    ap.add_argument("--robustness", default=None, help="robustness_3seed_summary_full.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    F1, DROP = load_inputs(a.summary, a.robustness)
    rng = np.random.default_rng(20260407)
    models = list(F1)
    rows, res = [], {}
    for sc, (b, wa, wr) in SCENARIOS.items():
        pts, sds, cis, draws = {}, {}, {}, {}
        for m in models:
            f, sd1 = F1[m][b]
            d, ds = DROP[m]
            pt = wa * f * 100 + wr * (100 - d)
            sd = np.sqrt((wa * 100 * sd1) ** 2 + (wr * ds) ** 2)
            sem = sd / np.sqrt(3)
            tf, td = rng.standard_t(2, a.draws), rng.standard_t(2, a.draws)
            draws[m] = pt + (wa * 100 * sd1 / np.sqrt(3)) * tf - (wr * ds / np.sqrt(3)) * td
            pts[m], sds[m], cis[m] = pt, sd, (pt - 4.303 * sem, pt + 4.303 * sem)
        top = np.vstack([draws[m] for m in models]).argmax(0)
        for i, m in enumerate(models):
            rows.append(dict(scenario=sc, model=m, drs=round(pts[m], 2), sd_single_run=round(sds[m], 2),
                             ci95_lo=round(cis[m][0], 1), ci95_hi=round(cis[m][1], 1),
                             p_rank1=round(float((top == i).mean()), 3)))
        res[sc] = rows[-len(models):]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "drs_uncertainty.csv"), index=False)

    L = [r"\begin{table}[ht]", r"\centering", r"\small", r"\begin{tabular}{llrrcr}", r"\toprule",
         r"\textbf{Scenario} & \textbf{Model} & \textbf{DRS} & \textbf{SD} & \textbf{95\% interval} & \textbf{P(rank 1)} \\",
         r"\midrule"]
    for sc in SCENARIOS:
        sub = df[df.scenario == sc].sort_values("drs", ascending=False)
        for i, (_, r) in enumerate(sub.iterrows()):
            first = SCEN_NAME[sc] if i == 0 else ""
            L.append(f"{first} & {r.model} & {r.drs:.1f} & {r.sd_single_run:.2f} & "
                     f"[{r.ci95_lo:.1f}, {r.ci95_hi:.1f}] & {r.p_rank1:.2f} \\\\")
        if sc != "C":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\label{tab:drs_uncertainty}Seed-level uncertainty of the Deployment Risk Score. DRS is computed "
          r"from the three-seed mean F1 at the scenario's label budget and the three-seed mean Avg.\ Drop (\%) of the "
          r"100\%-label models. SD: standard deviation of a single-run DRS (first-order propagation, sample SDs, "
          r"independent F1 and Avg.\ Drop). 95\% interval: DRS $\pm$ 4.30 standard errors ($t$, two degrees of freedom). "
          r"P(rank 1): share of 20{,}000 Monte-Carlo draws of the true-mean DRS (heavy-tailed $t_2$, scaled by the "
          r"standard errors) in which the model scores highest. Only training-seed variation is represented; "
          r"test-set sampling, the labelled-subset draw, the weights and pretraining-run variability (the SimCLR and "
          r"MAE encoders were pretrained once) are not. With three seeds these quantities are themselves uncertain.}",
          r"\end{table}"]
    open(os.path.join(a.out, "drs_uncertainty.tex"), "w").write("\n".join(L) + "\n")

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nWrote 2 files to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
