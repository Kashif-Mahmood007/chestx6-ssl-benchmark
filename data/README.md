# Data

This repository does **not** include the dataset images used in the
experiments. Following the loss of ChestX6's originally cited hosting
location during revision, we independently verified image-level
provenance -- SHA-256 exact matching, perceptual hashing, and
deep-embedding corroboration -- for every image, and this repository links
directly to the verified original sources rather than hosting a derived
copy, so that citation and reuse credit goes to their original creators.

## ChestX6

ChestX6 is a six-class chest X-ray benchmark (17,988 images,
post-deduplication) compiled from two publicly available sources:

| Classes | Images | Source | Link |
|---|---|---|---|
| Covid-19, Normal, Pneumonia-Bacterial, Pneumonia-Viral, Emphysema | 14,765 (82.1%) | "Dataset (Covid-Bacterial-Viral-Normal-Emphysema)" (Minh Nhat, Kaggle) | https://www.kaggle.com/datasets/minhnhat232/dataset-covid-bacterial-viral-normal-emphysema |
| Tuberculosis | 3,173 (17.6%) | Tuberculosis (TB) Chest X-ray Database (Rahman et al., 2020, *IEEE Access*) | https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset |

A small residual (0.3%) also matched two further public datasets,
consistent with upstream reuse within the primary source rather than an
independent contribution:

| Dataset | Link |
|---|---|
| Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays), V4 (Sait et al., 2020) | https://doi.org/10.17632/9xkhgts2s6.4 |
| COVID-19 Radiography Database (Chowdhury et al., 2020, *IEEE Access*) | https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database |

**Per-class provenance, verification methodology, and confidence tiers**
are documented in full in `notebooks/NB0 - Dataset Provenance
Verification.ipynb` and in the manuscript (Table: per-class provenance).

Prior to deduplication, 18,036 images were originally collected across
these sources (Covid-19: 3,017; Normal: 3,271; Pneumonia-Bacterial: 3,000;
Pneumonia-Viral: 3,013; Emphysema: 2,550; Tuberculosis: 3,185). MD5-based
deduplication removed 48 duplicate images, yielding the 17,988-image
experimental set used throughout this study.

## Fixed Splits and Checksums

This repository does not include split files directly -- they are
maintained on Kaggle as the single source of truth, to avoid two copies
drifting out of sync. The Kaggle repository contains:

* Train/validation/test split indices
* Label-budget subset indices
* MD5 checksums for dataset verification

**Source:**
https://www.kaggle.com/datasets/kashif03371733/chestx6-ssl-benchmark-splits-and-checksums

---

## ChestMNIST (Cross-Dataset Evaluation)

Cross-dataset evaluation is performed using **ChestMNIST**, a **22,433-image** binary chest X-ray dataset derived from NIH ChestX-ray14 and distributed as part of **MedMNIST v2**.

The dataset is downloaded automatically through the `medmnist` Python package.

**Source:**
https://medmnist.com

---

## Citing the Data

If you use ChestX6, please cite the two verified source datasets:

```bibtex
@misc{minhnhat2023chestx6base,
  author       = {{Minh Nhat}},
  title        = {Dataset (Covid-Bacterial-Viral-Normal-Emphysema)},
  year         = {2023},
  howpublished = {Kaggle},
  url          = {https://www.kaggle.com/datasets/minhnhat232/dataset-covid-bacterial-viral-normal-emphysema}
}

@article{rahman2020tb,
  author  = {Rahman, Tawsifur and Khandakar, Amith and Kadir, Muhammad Abdul and Islam, Khandaker Reajul and Islam, Khandaker Farhat and Mazhar, Rashid and Hamid, Tahir and Islam, Mohammad Tariqul and Kashem, Saad and Mahbub, Zaid Bin and Ayari, Mohamed Arselene and Chowdhury, Muhammad E. H.},
  title   = {Reliable Tuberculosis Detection Using Chest X-Ray With Deep Learning, Segmentation and Visualization},
  journal = {IEEE Access},
  volume  = {8},
  pages   = {191586--191601},
  year    = {2020},
  doi     = {10.1109/ACCESS.2020.3031384}
}
```