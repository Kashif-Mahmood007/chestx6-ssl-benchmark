# Checkpoints

The trained model checkpoints used in this study are hosted on Hugging Face rather than in this repository because their total size (~4 GB) exceeds the practical size for a GitHub repository.

**Hugging Face Repository**

```text
https://huggingface.co/Kashif-Mahmood007/chestx6-ssl-checkpoints
```

The repository contains:

* **`supervised/`**

  * ResNet50 (Scratch)
  * ResNet50 (ImageNet)
  * EfficientNet-B0
  * MobileViT-XS
  * ViT-S/16 (Scratch)
  * Models trained using **10%**, **20%**, and **100%** labeled data
  * 3 fixed random seeds: 42, 123, and 456

* **`ssl/`**

  * SimCLR pretrained encoder
  * MAE pretrained encoder

* **`ssl_trainonly_data/`**
  * SimCLR pretrained encoder (Only Training Data)
  * MAE pretrained encoder (Only Training Data)

* **`finetuned/`**

  * SimCLR fine-tuned models
  * MAE fine-tuned models
  * Models fine-tuned using **10%**, **20%**, and **100%** labeled data
  * **3 random seeds** for each training configuration

## Checkpoint File Formats

The checkpoint files are stored in the following formats:

| Directory     | File Format | Description                                                     |
| ------------- | ----------- | --------------------------------------------------------------- |
| `supervised/` | `.pt`       | Fully supervised model checkpoints                              |
| `ssl/`        | `.pth`      | Self-supervised pretrained encoder checkpoints (SimCLR and MAE) |
| `ssl_trainonly_data/`        | `.pth`      | Training-partition-only self-supervised encoder checkpoints |
| `finetuned/`  | `.pt`       | Fine-tuned self-supervised model checkpoints                    |

The `.pt` and `.pth` extensions are both standard PyTorch checkpoint formats. Refer to the notebooks in the `notebooks/` directory for the appropriate loading procedure (`load_state_dict`) for each model architecture.

After downloading, preserve the original directory structure so that the notebooks can locate the checkpoint files without modification.


## Paper Version

* The exact Hugging Face revision used for the paper is: 1764b2409e06a91e81eda0f8bfe0c8ae4e199a38
* Hugging Face DOI: https://doi.org/10.57967/hf/10799
* The code and results used in the paper are archived at Zenodo as GitHub release v1.0.0-scirep: https://doi.org/10.5281/zenodo.23200410
