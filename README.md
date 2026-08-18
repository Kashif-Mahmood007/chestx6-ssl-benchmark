# ChestX6-SSL-Benchmark

Code accompanying the manuscript:

**"A Deployment Risk Score Framework for Self-Supervised Chest X-Ray Classification: Calibrated Multi-Objective Evaluation Under Annotation Scarcity and Scanner Heterogeneity"**

**Authors:** Kashif Mahmood, Romana Aziz, Muhammad Ramzan, Mahwish Ilyas, and Ala Saleh Alluhaidan

This repository contains the code, experimental configurations, and supplementary resources associated with the above manuscript. The repository has been made publicly available to support reproducibility and the peer-review process.

The study compares contrastive self-supervised learning (**SimCLR**) and reconstruction-based self-supervised learning (**MAE**) against four supervised baselines using the **ChestX6** dataset. Evaluation includes:

* Label efficiency
* Corruption robustness
* Model calibration
* Cross-dataset transfer to ChestMNIST
* Deployment Risk Score (DRS), an exploratory multi-objective decision-support metric for scenario-specific model selection

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
│   ├── NB4 - Domain Generalization & Calibration
│   ├── NB5 - Advanced analysis & figure generation
│   └── NB6 - Cross-dataset transfer (ChestMNIST)
│
├── figures/            # All manuscript figures (300 DPI)
├── results/            # CSV/JSON result files
│
├── checkpoints/
│   └── README.md       # Download links for pretrained models
│
├── data/
│   └── README.md       # Dataset sources and download instructions
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Models Evaluated

| Model               | Pretraining                      | Parameters |
| ------------------- | -------------------------------- | ---------: |
| ResNet50 (Scratch)  | None                             |     25.6 M |
| ResNet50 (ImageNet) | Supervised ImageNet              |     25.6 M |
| EfficientNet-B0     | Supervised ImageNet              |      5.3 M |
| MobileViT-XS        | Supervised ImageNet              |      5.6 M |
| SimCLR + ResNet50   | Contrastive SSL (ChestX6)        |     25.6 M |
| MAE + ViT-S/16      | Masked Autoencoder SSL (ChestX6) |     22.1 M |

---

# Installation

Clone the repository and install the required packages.

```bash
git clone https://github.com/Kashif-Mahmood007/ChestX6-SSL-Benchmark.git
cd ChestX6-SSL-Benchmark
pip install -r requirements.txt
```

---

# Datasets

This project uses two publicly available datasets.

### ChestX6

ChestX6 is a compiled six-class chest X-ray benchmark. It is not hosted in
this repository or under a dedicated dataset DOI — we independently
verified, at the image level, that it draws from two publicly available
sources, and link directly to both so that citation and reuse credit goes
to the original creators. Full provenance, per-class counts, and download
instructions are in `data/README.md`.

### ChestMNIST

ChestMNIST is automatically downloaded through the `medmnist` package during evaluation.

---

# Reproducing the Results

1. Install the required dependencies.
2. Download the source images listed in `data/README.md` and assemble ChestX6 following the class mapping given there.
3. Download the fixed train/validation/test splits and MD5 checksums from the Kaggle repository linked in `data/README.md`.
4. Download pretrained checkpoints from the location specified in `checkpoints/README.md`, or train the models from scratch using the notebooks.
5. Execute the notebooks in order to reproduce the complete experimental pipeline and manuscript figures.

---

# Environment

Experiments were conducted using:

* Python 3.x
* PyTorch
* torchvision
* timm
* medmnist
* NumPy
* pandas
* scikit-learn

Training and evaluation were performed using the free-tier **Google Colab** environment with an **NVIDIA T4 GPU**.

Exact package versions are provided in `requirements.txt`.

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{[citekey]2026,
  title  = {A Deployment Risk Score Framework for Self-Supervised Chest X-Ray Classification: Calibrated Multi-Objective Evaluation Under Annotation Scarcity and Scanner Heterogeneity},
  author = {Kashif Mahmood and Romana Aziz and Muhammad Ramzan and Mahwish Ilyas and Ala Saleh Alluhaidan},
  note   = {Manuscript under revision},
  year   = {2026}
}
```

### Dataset Citation

If you use the ChestX6 class compilation (not just this code), please also
cite the two source datasets it independently verifies to. Full citations
are in `data/README.md`.

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for details.