# Split files of ChestX6 (17,988 images)

The JSON index files are the authoritative definition of the splits. Each is a list of row positions (0-based) in `md5_checksums.csv` (17,988 rows, deduplicated, one row per image).

| File | Content |
|---|---|
| train_indices.json, val_indices.json, test_indices.json | original stratified split (70/10/20, seed 42): 12,591 / 1,799 / 3,598 images, used for all main results |
| subset_10pct_seed42.json, subset_20pct_seed42.json | labelled subsets of the training split (10% and 20% label budgets) |
| md5_checksums.csv | filename, class, MD5 and split of the 17,988 images (Tuberculosis files are listed by their raw .jpeg names) |
| train_indices_tbcorrected.json, val_indices_tbcorrected.json, test_indices_tbcorrected.json | group-respecting split of the leakage sensitivity analysis: only the Tuberculosis class is reassigned, all derivatives of one original acquisition stay in one partition; every other class keeps its original membership |
| tb_manifest.csv | the 3,173 Tuberculosis images with their original acquisition (acq_id, 700 acquisitions), their split in the original split and in the group-respecting split |
