# ChestX6-SSL-Benchmark

[![DOI](https://zenodo.org/badge/1324454217.svg)](https://zenodo.org/badge/latestdoi/1324454217)

Code accompanying the manuscript:

**"A Deployment Risk Score Framework for Self-Supervised Chest X-Ray Classification: Calibrated Multi-Objective Evaluation Under Annotation Scarcity and Distribution Shift"**

**Authors:** Kashif Mahmood, Romana Aziz, Muhammad Ramzan, Mahwish Ilyas, and Ala Saleh Alluhaidan

This repository contains the code, experimental configurations, and supplementary resources associated with the above manuscript. The repository has been made publicly available to support reproducibility and the peer-review process.

The study compares two complete self-supervised pipelines, a contrastive one (**SimCLR** + ResNet50) and a masked-autoencoder-style one (**MAE** + ViT-S/16; see the [note on what its pretraining trained](#important-note-on-the-mae--vit-s16-model)), against five supervised baselines (including an architecture-matched ViT-S/16 trained from scratch) using the **ChestX6** dataset. The two pipelines differ in backbone, objective, epochs and number of updates and are not compute-matched, so the comparison is between pipelines, not a controlled test of the pretraining objective. Evaluation includes:

* Label efficiency (10%, 20% and 100% annotation budgets)
* Robustness to eight synthetic image perturbations
* Model calibration (ECE, Brier score, NLL, temperature scaling)
* Cross-dataset transfer to ChestMNIST (external-dataset shift; binary labels)
* Class-specific sensitivity/specificity and confusion matrices with bootstrap confidence intervals
* Evaluation-independence audits: SSL pretraining restricted to the training partition, Tuberculosis augmentation-lineage leakage, source-identity confound, and a ChestX6–ChestMNIST overlap check
* Deployment Risk Score (DRS), an exploratory multi-objective decision-support metric for scenario-specific model selection (not a validated clinical instrument)

> **Research use only.** This repository supports a retrospective research comparison on public datasets. It is **not a medical device**, has not been validated for clinical use, and must not be used to inform diagnosis or treatment.

---

# Archived versions used in the paper

The links below point to frozen snapshots, not to the latest state of each repository. Please cite and download these versions to reproduce the paper.

| Resource | Archived version used in the paper |
| -------- | ---------------------------------- |
| Code, results and split files (this repository) | Release `v1.0.0-scirep`, archived at Zenodo: https://doi.org/10.5281/zenodo.23202334 |
| Model checkpoints (Hugging Face) | https://huggingface.co/Kashif-Mahmood007/chestx6-ssl-checkpoints, revision `1764b2409e06a91e81eda0f8bfe0c8ae4e199a38` (DOI: https://doi.org/10.57967/hf/10799) |
| Split indices and MD5 checksums (Kaggle) | Version 3, DOI: https://doi.org/10.34740/KAGGLE/DSV/20409572 |

The split-index files, the MD5 checksums, the group-respecting Tuberculosis split and the Tuberculosis manifest are stored in `data/splits/` of this repository, so they are part of the Zenodo archive of the code. The Kaggle dataset repeats the original split and checksum files. The `scripts/` folder, which rebuilds the supplementary tables, is part of the same archive.

---

# Repository Structure

```text
chestx6-ssl-benchmark/
│
├── notebooks/          # End-to-end training and analysis notebooks
│   ├── NB0 - Dataset Provenance Verification
│   ├── NB1 - Dataset Audit & Official Split
│   ├── NB2 - Supervised training
│   ├── NB3 - SSL Pretraining & Fine-tuning
│   ├── NB3 - SSL Pretraining & Fine-tuning (Train-only Data)
│   ├── NB4 - Robustness (Synthetic Perturbations) & Calibration
│   ├── NB5 - Advanced analysis & figure generation
│   └── NB6 - Cross-dataset transfer (ChestMNIST)
│
├── figures/            # All manuscript figures (300 DPI)
├── results/            # CSV/JSON result files (one JSON per model x budget x seed), plus
│                       # per_seed_results.csv and effect_sizes.csv (data of Supplementary Tables S1 and S3)
│
├── scripts/            # Post-hoc analysis scripts for the supplementary tables
│   ├── make_per_seed_results.py          # per-seed table, SDs, effect sizes, exact p-values (Tables S1, S3)
│   ├── make_supp_tables_S2_S4.py         # absolute robustness F1 and AUC intervals (Tables S2, S4)
│   ├── make_drs_uncertainty.py           # seed-level uncertainty of the DRS (Table S5)
│   ├── build_supplementary.py            # runs the three table scripts and compiles Tables S1-S5 into one PDF
│   ├── count_tb_acquisitions.py          # counts original Tuberculosis acquisitions per partition (Section 5.3, Limitation 2)
│   ├── check_mae_gradients.py            # shows which ViT parameters receive a gradient in MAE pretraining
│   └── NB3_MAE_pretrain_corrected_cell.py   # corrected MAE pretraining (NOT used for the paper's results)
│
├── checkpoints/
│   └── README.md       # Download links for pretrained models (pinned revision)
│
├── data/
│   ├── README.md       # Dataset sources, licences and download instructions
│   └── splits/         # Split index files (JSON), MD5 checksums, group-respecting Tuberculosis split
│                       # and Tuberculosis manifest; see data/splits/README.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Models Evaluated

| Model               | Pretraining                                  | Parameters |
| ------------------- | -------------------------------------------- | ---------: |
| ResNet50 (Scratch)  | None                                         |     25.6 M |
| ResNet50 (ImageNet) | Supervised ImageNet                          |     25.6 M |
| EfficientNet-B0     | Supervised ImageNet                          |      5.3 M |
| MobileViT-XS        | Supervised ImageNet                          |      5.6 M |
| ViT-S/16 (Scratch)  | None (architecture-matched control for MAE)  |     21.7 M |
| SimCLR + ResNet50   | Contrastive SSL (ChestX6)                    |     25.6 M |
| MAE + ViT-S/16      | MAE-style SSL (ChestX6); only the patch-embedding layer was pretrained (see note below) |     22.1 M |

SimCLR and MAE were each pretrained once on ChestX6 images and then fine-tuned with three seeds (42, 123, 456). All seven models were trained under one fixed protocol; no hyperparameter search was performed. A second set of SimCLR and MAE encoders pretrained on the training partition only is used for the evaluation-independence sensitivity analysis (train-only variant of NB3 in `notebooks/`; per-run results in `results/`).

### Important note on the "MAE + ViT-S/16" model

In the pretraining code used for the paper (`notebooks/NB3`, Cell 4) the visible patch embeddings produced by `encoder.patch_embed` were passed directly to the decoder. The ViT's transformer blocks, class token, positional embedding and final norm were therefore never part of the pretraining forward pass and were not updated. The "MAE + ViT-S/16" checkpoints contain a **pretrained patch-embedding layer and randomly initialised transformer blocks**, exactly as in ViT-S/16 (Scratch). The MAE results in the paper therefore do **not** test masked-reconstruction pretraining of a full ViT encoder; they describe the pipeline as it was run.

A corrected implementation that trains the full encoder is provided in `scripts/NB3_MAE_pretrain_corrected_cell.py`. **It was not used to produce any result in the paper.** `scripts/check_mae_gradients.py` shows the difference in a few seconds on a CPU (no data or GPU needed).

For SimCLR, the NT-Xent loss was computed inside micro-batches of 64 images (126 negatives per anchor), not over the effective batch of 256 obtained by gradient accumulation.

---

# Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/Kashif-Mahmood007/ChestX6-SSL-Benchmark.git
cd ChestX6-SSL-Benchmark
# The pinned torch/torchvision builds (+cu128) are not on PyPI; the extra index is required.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The pins in `requirements.txt` are the versions listed under [Environment](#environment). For a machine without an NVIDIA GPU, install the matching CPU builds of `torch` and `torchvision` from https://download.pytorch.org/whl/cpu instead.

---

# Datasets

This project uses two publicly available datasets. **No image data is stored in this repository.**

### ChestX6

ChestX6 is a compiled six-class chest X-ray benchmark (18,036 images collected; 17,988 after removing 48 exact duplicates by MD5 audit). It is not hosted in this repository or under a dedicated dataset DOI — we independently verified, at the image level, that it draws from two publicly available sources, and link directly to both so that citation and reuse credit goes to the original creators:

* Covid-19, Normal, Pneumonia-Bacterial, Pneumonia-Viral and Emphysema (82.1%): https://www.kaggle.com/datasets/minhnhat232/dataset-covid-bacterial-viral-normal-emphysema
* Tuberculosis (17.6%): https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

The remaining 0.3% matched the curated dataset of Sait et al. and the COVID-19 Radiography Database. Full provenance, per-class counts, licences, and download instructions are in `data/README.md`. The fixed splits (70/10/20 stratified, seed 42; 12,591 / 1,799 / 3,598 images) and MD5 checksums of all retained images are in `data/splits/`.

### ChestMNIST

ChestMNIST (28×28 release, derived from NIH ChestX-ray14) is automatically downloaded through the `medmnist` package during evaluation. It is used only as an external test set; its binary labels differ from ChestX6's six-class scheme.

---

# Reproducing the Results

1. Install the required dependencies.
2. Download the source images listed in `data/README.md` and assemble ChestX6 following the class mapping given there.
3. Use the fixed train/validation/test splits and MD5 checksums in `data/splits/` (or the archived Kaggle version listed above). The index files are lists of row positions (0-based) in `md5_checksums.csv`. The group-respecting Tuberculosis split of the leakage sensitivity analysis is given by `train_indices_tbcorrected.json`, `val_indices_tbcorrected.json` and `test_indices_tbcorrected.json` (only the Tuberculosis images are reassigned), and `tb_manifest.csv` lists every Tuberculosis image with its original acquisition and its partition in both splits.
4. Download the checkpoints at the pinned revision given in `checkpoints/README.md`, or train the models from scratch using the notebooks. SimCLR and MAE pretraining each ran once; only fine-tuning was repeated across seeds.
5. Execute the notebooks in order to reproduce the complete experimental pipeline and manuscript figures.
6. To rebuild the supplementary tables from the per-run JSON files in `results/`, run `python scripts/make_per_seed_results.py results --out out/` (Tables S1 and S3, per-seed CSV, summary with sample and population standard deviations), `python scripts/make_supp_tables_S2_S4.py ...` (Tables S2 and S4) and `python scripts/make_drs_uncertainty.py --out out/` (Table S5); each script's docstring gives the exact command. To rebuild all five tables and the two data files in one step, run `python scripts/build_supplementary.py --results results --scripts scripts --out supplementary_upload` (needs `pdflatex`). `python scripts/count_tb_acquisitions.py data/splits/tb_manifest.csv` reproduces the acquisition counts of Section 5.3 (Limitation 2) and checks them against the totals stated there.

Exact numerical reproducibility across different hardware or library versions is not guaranteed: CUDA-level bit-exact determinism was not enforced beyond the random seeds used for data splitting and initialization. Results are expected to fall within the seed-to-seed variation reported in the paper.

---

# Environment

Experiments were conducted using:

* Python 3.13
* CUDA 12.8
* torch 2.11.0+cu128, torchvision 0.26.0+cu128
* timm 1.0.29
* numpy 2.1.3, pandas 2.2.3, scipy 1.16.3, scikit-learn 1.6.1
* medmnist 3.0.2, Pillow 11.3.0, kornia 0.8.3, grad-cam 1.5.7
* matplotlib 3.10.0, seaborn 0.13.2, umap-learn 0.5.12, tqdm 4.67.3

Training and evaluation were performed using the free-tier **Google Colab** environment with an **NVIDIA T4 GPU**.

These are the exact versions recorded from the environment that produced the results; `requirements.txt` pins the same versions.

---

# Citation

If you use this repository in your research, please cite the paper and the archived code release:

```bibtex
@article{mahmood2026chestx6,
  title  = {A Deployment Risk Score Framework for Self-Supervised Chest X-Ray Classification: Calibrated Multi-Objective Evaluation Under Annotation Scarcity and Distribution Shift},
  author = {Kashif Mahmood and Romana Aziz and Muhammad Ramzan and Mahwish Ilyas and Ala Saleh Alluhaidan},
  note   = {Manuscript under revision at Scientific Reports},
  year   = {2026}
}

@software{mahmood2026chestx6code,
  title   = {ChestX6-SSL-Benchmark: code, split files and results for the manuscript above},
  author  = {Kashif Mahmood and Romana Aziz and Muhammad Ramzan and Mahwish Ilyas and Ala Saleh Alluhaidan},
  version = {v1.0.0-scirep},
  year    = {2026},
  doi     = {10.5281/zenodo.23202334},
  url     = {https://github.com/Kashif-Mahmood007/ChestX6-SSL-Benchmark}
}
```

### Dataset Citation

If you use the ChestX6 class compilation (not just this code), please also cite the two source datasets it independently verifies to. Full citations are in `data/README.md`. If you use the external test set, please also cite MedMNIST v2 and acknowledge the NIH Clinical Center as the provider of the original ChestX-ray14 data.

---

# License

The **code** in this repository is released under the **MIT License** (see the `LICENSE` file). The split-index, checksum and manifest files in `data/splits/` are released under **CC BY 4.0**. These files and the model checkpoints contain **no image data**.

**Intended use.** This repository supports a retrospective research comparison on public datasets. It is **not a medical device**, has not been validated for clinical use, and must not be used to inform diagnosis or treatment. The Deployment Risk Score is an exploratory research tool.

**Data.** No source images are redistributed. The source datasets remain under the terms set by their providers; see the licence table in `data/README.md`. You must obtain the images from the original providers and comply with their terms. Checkpoints were trained on those data; check that your intended use is compatible with the source terms.
