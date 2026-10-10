#!/usr/bin/env python3
"""
check_mae_gradients.py -- verify, in 20 seconds on a CPU, which parts of the ViT receive a gradient in MAE pretraining.

  ORIGINAL cell (NB3 CELL 4):   tokens = encoder.patch_embed(imgs)  ->  decoder
  CORRECTED cell:               patch_embed + pos_embed -> visible patches -> cls token -> ALL blocks -> norm -> decoder

Run:   python check_mae_gradients.py          (needs torch and timm only; no data, no GPU, no download)

Expected output: with the ORIGINAL pipeline only patch_embed.proj.weight has a gradient (every transformer block,
the cls token, pos_embed and the final norm print "grad is None: True"); with the CORRECTED pipeline all of them
have one. Gradient = None means the optimiser never updates the parameter, so it stays at its random initialisation.
"""
import timm
import torch


def make_mask(B, n_patches=196, ratio=0.75):
    n_mask = int(n_patches * ratio)
    mask = torch.zeros(B, n_patches, dtype=torch.bool)
    for b in range(B):
        mask[b, torch.randperm(n_patches)[:n_mask]] = True
    return mask


def encode_visible(encoder, imgs, mask):
    B = imgs.shape[0]
    x = encoder.patch_embed(imgs) + encoder.pos_embed[:, 1:, :]
    n_vis = int((~mask[0]).sum())
    x = x[~mask].reshape(B, n_vis, -1)
    cls = (encoder.cls_token + encoder.pos_embed[:, :1, :]).expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    if hasattr(encoder, "norm_pre"):
        x = encoder.norm_pre(x)
    x = encoder.norm(encoder.blocks(x))
    return x[:, 1:, :]


def report(title, enc):
    probes = {"patch_embed.proj.weight": enc.patch_embed.proj.weight, "cls_token": enc.cls_token, "pos_embed": enc.pos_embed,
              "blocks.0.attn.qkv.weight": enc.blocks[0].attn.qkv.weight,
              "blocks.11.mlp.fc2.weight": enc.blocks[11].mlp.fc2.weight, "norm.weight": enc.norm.weight}
    print(f"\n{title}")
    for k, p in probes.items():
        print(f"   {k:28s} grad is None: {p.grad is None}")


torch.manual_seed(0)
imgs, mask = torch.randn(2, 3, 224, 224), make_mask(2)

enc_old = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
tok = enc_old.patch_embed(imgs)                       # what the ORIGINAL cell did
tok[~mask].reshape(2, 49, -1).pow(2).mean().backward()
report("ORIGINAL pipeline (patch_embed only):", enc_old)

enc_new = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
encode_visible(enc_new, imgs, mask).pow(2).mean().backward()
report("CORRECTED pipeline (all blocks):", enc_new)
