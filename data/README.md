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

### Per-class provenance, verification methodology, and confidence tiers
Per-class provenance, verification methodology, and confidence tiers are documented in full in notebooks/NB0 - Dataset Provenance Verification.ipynb and in the manuscript (Table: per-class provenance).

Prior to deduplication, 18,036 images were originally collected across these sources:
* Covid-19: 3,017
* Normal: 3,271
* Pneumonia-Bacterial: 3,000
* Pneumonia-Viral: 3,013
* Emphysema: 2,550
* Tuberculosis: 3,185

MD5-based deduplication removed 48 duplicate images, yielding the 17,988-image experimental set used throughout this study.

## Fixed Splits and Checksums

The split files are stored in this repository in [`splits/`](splits/); the
file `splits/README.md` describes every file. The same eleven files are
archived on Kaggle (Version 4, below). They contain:

* Train/validation/test split indices of the original stratified split
  (70/10/20, seed 42), used for all main results
* Label-budget subset indices (10% and 20%)
* MD5 checksums, with the split of each image, for dataset verification
* The group-respecting Tuberculosis split used in the leakage sensitivity
  analysis, and `tb_manifest.csv`, which lists the 3,173 Tuberculosis images
  with their original acquisition and their partition in both splits

## Kaggle reproducibility package

### ChestX6: SSL Benchmark Splits & Checksums
**Version:** 4 (the version cited by the paper)
**DOI:** https://doi.org/10.34740/KAGGLE/DSV/20558883
**Dataset page:**
https://www.kaggle.com/datasets/kashif03371733/chestx6-ssl-benchmark-splits-and-checksums

The index, checksum and manifest files are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) License. This licence covers only those files and not any image data.

Version 3 should not be used: its `md5_checksums.csv` listed 18,036 rows (the 48 duplicate images had not been removed) and its split column disagreed with the JSON index files for 8,189 of the 17,988 images. The JSON index files of Version 3 were correct and are unchanged in Version 4.

---


## ChestMNIST (Cross-Dataset Evaluation)

Cross-dataset evaluation is performed using **ChestMNIST**, a **22,433-image** binary chest X-ray dataset derived from NIH ChestX-ray14 and distributed as part of **MedMNIST v2**.

The dataset is downloaded automatically through the `medmnist` Python package.

**Source:**
https://medmnist.com

---

## Licences and permitted use

No source image is redistributed by this repository. Each image remains under the terms set by its provider, including the terms of any upstream source that the provider compiles. Obtain the images from the original providers and comply with their terms. The released checkpoints were trained on these data, and their use is subject to the same terms. The terms below are those published by each provider (checked in October 2026).

| Source | Role | Terms as published by the provider |
|---|---|---|
| Minh Nhat Kaggle compilation (five classes, 82.1% of ChestX6) | training and testing | Kaggle licence "CC0: Public Domain", set by the compiler; its description states that it compiles images from Sait et al., NIH ChestX-ray14 and the COVID-QU dataset, whose own terms continue to apply |
| Rahman et al. Tuberculosis Chest X-ray Database (all Tuberculosis images, including the augmented derivatives) | training and testing | Kaggle licence "Data files © Original Authors"; the database aggregates images from the NLM (Montgomery and Shenzhen) collections, the Belarus tuberculosis dataset, the NIAID TB Portal and the RSNA pneumonia detection challenge, whose terms continue to apply |
| Sait et al., Curated Dataset V4 (Mendeley Data; residual 0.3%) | residual match | CC BY 4.0 (Mendeley Data record, DOI 10.17632/9xkhgts2s6.4); it collates 15 public datasets, whose terms also apply |
| COVID-19 Radiography Database (Kaggle; residual 0.3%) | residual match | Kaggle licence "Data files © Original Authors"; the page states academic, non-commercial use |
| NIH ChestX-ray14 (NIH Clinical Center) | origin of the external test set | images: users must link to the NIH download site (https://nihcc.app.box.com/v/ChestXray-NIHCC), cite Wang et al. (CVPR 2017) and acknowledge the NIH Clinical Center; the LLM-generated annotation files listed on that page (`nih_llm_annotations_*.csv`) are separately licensed CC BY-NC 4.0 because they derive from a fine-tuned language model, and we do not use them |
| MedMNIST v2 / ChestMNIST | external test set | collection CC BY 4.0 (the only exception is DermaMNIST), code Apache-2.0; "not intended for clinical use" |

*Augmented Tuberculosis subset.* Of the 3,173 Tuberculosis images, 2,482 are augmented derivatives of the 700 original images of the Rahman et al. database (their parent-child lineage was recovered from the file names); the other 691 matched source originals directly. We make no claim about the licence of the derivatives beyond that of the database they derive from, and we redistribute none of them.

*Our own releases.* The split-index, checksum and manifest files (`splits/` and Kaggle Version 4) are released under CC BY 4.0; the code, the per-run result files and the model checkpoints are released under the MIT licence. None of them contains image data.

---

## Citing the Data

If you use ChestX6, please cite the original source datasets listed above and the ChestX6 reproducibility package.


### Minh Nhat dataset
```bibtex
@misc{minhnhat2023chestx6base,
  author       = {{Minh Nhat}},
  title        = {Dataset (Covid-Bacterial-Viral-Normal-Emphysema)},
  year         = {2023},
  howpublished = {Kaggle},
  url          = {https://www.kaggle.com/datasets/minhnhat232/dataset-covid-bacterial-viral-normal-emphysema}
}

```

### Tuberculosis Chest X-ray Database
``` bibtex
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


### ChestX6 reproducibility package
``` bibtex
@misc{kashif_mahmood_romana_aziz_muhammad_ramzan_mahwish_ilyas_ala_saleh_alluhaidan_2026,
  title     = {ChestX6: SSL Benchmark Splits & Checksums},
  publisher = {Kaggle},
  year      = {2026},
  version   = {4},
  doi       = {10.34740/KAGGLE/DSV/20558883},
  url       = {https://www.kaggle.com/dsv/20558883},
  author    = {Kashif Mahmood and Romana Aziz and Muhammad Ramzan and Mahwish Ilyas and Ala Saleh Alluhaidan}
}

```