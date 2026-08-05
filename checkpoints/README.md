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
  * Models trained using **10%**, **20%**, and **100%** labeled data
  * **3 random seeds** for each training configuration

* **`ssl/`**

  * SimCLR pretrained encoder
  * MAE pretrained encoder

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
| `finetuned/`  | `.pt`       | Fine-tuned self-supervised model checkpoints                    |

The `.pt` and `.pth` extensions are both standard PyTorch checkpoint formats. Refer to the notebooks in the `notebooks/` directory for the appropriate loading procedure (`load_state_dict`) for each model architecture.

After downloading, preserve the original directory structure so that the notebooks can locate the checkpoint files without modification.
