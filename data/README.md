# Data

This repository does **not** include the datasets used in the experiments. Both datasets are publicly available from their original sources. To support reproducibility, this repository provides the fixed data splits used throughout the study.

## ChestX6

The experiments in this study use the **ChestX6: A Compiled Six-Class Chest X-Ray Classification Benchmark**, originally containing **18,036** chest X-ray images across six classes:

* COVID-19
* Emphysema
* Normal
* Pneumonia-Bacterial
* Pneumonia-Viral
* Tuberculosis

During dataset preparation, **48 duplicate images were identified and removed**, resulting in a final dataset of **17,988** unique images used in all experiments.

**Original dataset:**
https://doi.org/10.5281/zenodo.21880301

## Fixed Splits and Checksums

This repository uses predefined train, validation, and test splits together with fixed label-budget subsets for all experiments.

The accompanying dataset contains:

* Train/validation/test split indices
* Label-budget subset indices
* MD5 checksums for dataset verification

These files enable exact reproduction of all experiments reported in the manuscript.

**Source:**
https://www.kaggle.com/datasets/kashif03371733/chestx6-ssl-benchmark-splits-and-checksums

---

## ChestMNIST (Cross-Dataset Evaluation)

Cross-dataset evaluation is performed using **ChestMNIST**, a **22,433-image** binary chest X-ray dataset derived from NIH ChestX-ray14 and distributed as part of **MedMNIST v2**.

The dataset is downloaded automatically through the `medmnist` Python package.

**Source:**
https://medmnist.com
