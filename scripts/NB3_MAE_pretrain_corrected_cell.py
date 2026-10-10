# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 (CORRECTED) — MAE Pretraining (ViT-S/16)         ║
# ║  Replaces the original CELL 4 of NB3.                    ║
# ║                                                          ║
# ║  WHAT WAS WRONG IN THE ORIGINAL CELL                     ║
# ║    tokens = encoder.patch_embed(imgs) was passed straight║
# ║    to the decoder: the ViT blocks, cls token, pos_embed  ║
# ║    and final norm never ran, so only the patch embedding ║
# ║    (+ the decoder) was trained.                          ║
# ║  WHAT THIS CELL DOES                                     ║
# ║    He et al. (2022): patch embed + pos_embed -> keep the ║
# ║    visible 25% -> cls token -> ALL 12 ViT blocks -> norm ║
# ║    -> light decoder with mask tokens -> MSE on masked    ║
# ║    patches (optional per-patch normalised targets).      ║
# ║  Checkpoints get NEW names (…_fullencoder_…) so that the ║
# ║  old patch-embedding-only encoder is not overwritten;    ║
# ║  point the fine-tuning cells to the new file.            ║
# ╚══════════════════════════════════════════════════════════╝

def encode_visible(encoder, imgs, mask):
    """
    MAE encoder forward pass on the VISIBLE patches only (He et al., 2022).

    imgs : (B, 3, 224, 224);  mask : BoolTensor (B, n_patches), True = masked (every row has the same count)
    Returns : (B, N_vis, feat_dim) encoded visible tokens, in ascending patch order within each image.
    The class token is run through the blocks with the visible tokens and dropped afterwards.
    """
    B = imgs.shape[0]
    x = encoder.patch_embed(imgs)                              # (B, 196, D)
    x = x + encoder.pos_embed[:, 1:, :].to(x.dtype)            # positions of the 196 patches
    D = x.shape[-1]
    N_vis = int((~mask[0]).sum().item())
    x = x[~mask].reshape(B, N_vis, D)                          # visible tokens only
    cls = (encoder.cls_token + encoder.pos_embed[:, :1, :]).to(x.dtype).expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)                             # (B, 1 + N_vis, D)
    if hasattr(encoder, 'norm_pre'):
        x = encoder.norm_pre(x)
    x = encoder.blocks(x)                                      # <-- the transformer blocks (never ran before)
    x = encoder.norm(x)
    return x[:, 1:, :]                                         # drop cls -> (B, N_vis, D)


def pretrain_mae(epochs=150, batch_size=64, accum_steps=2,
                 mask_ratio=0.75, norm_pix_loss=True):
    """
    MAE (Masked Autoencoder) pretraining with ViT-S/16.
    Reconstructs randomly masked 75% of image patches.

    Key settings:
      epochs=100    : longer training improves ViT representations
      accum_steps=2 : actual batch=32, safe on T4 15GB
      mask_ratio=0.75 : standard MAE setting (He et al. 2022)
      AMP enabled   : reconstruction loss is fp16-safe unlike NT-Xent

    Disconnect safety:
      Resume checkpoint saved every 5 epochs (~25 min safety window)
      On reconnect: re-run Cell 2b then this cell — resumes automatically

    Parameters
    ----------
    epochs      : int   pretraining epochs
    batch_size  : int   effective batch size
    accum_steps : int   gradient accumulation steps
    mask_ratio  : float fraction of patches to mask (default 0.75)
    norm_pix_loss : bool  normalise each target patch by its own mean/std (He et al. 2022, better features)
    """
    final_ckpt  = f'{SSL_CKPT_DIR}/mae_vits16_fullencoder_final.pth'
    resume_ckpt = f'{SSL_CKPT_DIR}/mae_vits16_fullencoder_resume.pt'

    # ── Skip-if-done ─────────────────────────────────────────
    if os.path.exists(final_ckpt):
        print('MAE already complete.')
        return final_ckpt

    actual_batch = batch_size // accum_steps

    print(f'\n{"="*60}')
    print(f'  MAE Pretraining (ViT-S/16, mask_ratio={mask_ratio})')
    print(f'  epochs={epochs}  effective_batch={batch_size}')
    print(f'  actual_batch_per_step={actual_batch}  accum_steps={accum_steps}')
    print(f'  Checkpoint every 5 epochs (~25 min safety window)')
    print(f'{"="*60}')

    # ── MAE Encoder (ViT-S/16) ────────────────────────────────
    encoder   = timm.create_model('vit_small_patch16_224',
                                  pretrained=False, num_classes=0)
    feat_dim  = encoder.num_features   # 384 for ViT-S
    patch_dim = 16 * 16 * 3            # 768 raw pixel values per patch
    n_patches = (224 // 16) ** 2       # 196 patches total

    # ── MAE Decoder (lightweight transformer) ─────────────────
    decoder_dim   = 512
    decoder_depth = 4
    decoder_heads = 16

    class MAEDecoder(nn.Module):
        """
        Lightweight transformer decoder for patch reconstruction.

        Fixed from original:
          - Per-batch mask assignment (not single-index broadcast)
          - enable_nested_tensor=False suppresses UserWarning
          - mask passed as (B, n_patches) BoolTensor directly
        """
        def __init__(self):
            super().__init__()
            self.proj       = nn.Linear(feat_dim, decoder_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.pos_embed  = nn.Parameter(
                torch.randn(1, n_patches, decoder_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer,
                num_layers=decoder_depth,
                enable_nested_tensor=False,   # suppresses norm_first warning
            )
            self.head = nn.Linear(decoder_dim, patch_dim)

        def forward(self, visible_tokens, mask, n_patches):
            """
            Parameters
            ----------
            visible_tokens : Tensor (B, N_vis, feat_dim)
                             Encoded visible patch tokens from encoder
            mask           : BoolTensor (B, n_patches)
                             True = masked (hidden), False = visible
            n_patches      : int  total number of patches (196)

            Returns
            -------
            Tensor (B, n_patches, patch_dim)
                Reconstructed pixel values for all patches
            """
            B = visible_tokens.shape[0]

            # Project visible tokens to decoder dimension
            x = self.proj(visible_tokens)   # (B, N_vis, decoder_dim)

            # Start with learnable mask tokens for all positions
            # AMP may produce x in fp16 while mask_token stays fp32
            full = self.mask_token.expand(B, n_patches, -1).clone().to(x.dtype)

            # Fill the visible positions (row-major order = ascending patch index per image,
            # the same order in which encode_visible() returns the tokens)
            full[~mask] = x.reshape(-1, x.shape[-1])

            # Add positional embedding and pass through transformer
            full = full + self.pos_embed.to(x.dtype)              # (B, n_patches, decoder_dim)
            out  = self.transformer(full)              # (B, n_patches, decoder_dim)
            return self.head(out)                      # (B, n_patches, patch_dim)

    encoder = encoder.to(DEVICE)
    decoder = MAEDecoder().to(DEVICE)

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1.5e-4, weight_decay=0.05
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(DEVICE.type,
                                     enabled=(DEVICE.type == 'cuda'))

    # ── MAE Dataset (single view — no two-crop needed) ────────
    from torchvision import transforms as T
    mae_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class MAEDataset(Dataset):
        """Single-view dataset for MAE pretraining (no labels used)."""
        def __init__(self, dataframe, transform):
            self.paths     = dataframe['image_path'].tolist()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            from PIL import Image
            try:
                img = Image.open(self.paths[i]).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224))
            return self.transform(img)

    dataset = MAEDataset(df, mae_transform)
    loader  = DataLoader(
        dataset, batch_size=actual_batch,
        shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    # ── Resume if checkpoint exists ───────────────────────────
    start_epoch  = 1
    loss_history = []

    if os.path.exists(resume_ckpt):
        print('  [RESUME] MAE resume checkpoint found.')
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        encoder.load_state_dict(ckpt['encoder_state'])
        decoder.load_state_dict(ckpt['decoder_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch  = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f'  Resuming from epoch {start_epoch}/{epochs}')
        if loss_history:
            print(f'  Best loss so far: {min(loss_history):.4f}')

    # ── Patchify helper ───────────────────────────────────────
    def patchify(imgs):
        """
        Convert images to patch tokens.
        imgs : Tensor (B, 3, 224, 224)
        Returns : Tensor (B, 196, 768)
        """
        B, C, H, W = imgs.shape
        p    = 16
        h    = w = H // p
        x    = imgs.reshape(B, C, h, p, w, p)
        x    = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
        return x

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches  = 0
        optimizer.zero_grad()
        t0 = time.time()

        bar = tqdm(loader, desc=f'  MAE ep{epoch}/{epochs}',
                   leave=False, mininterval=5.0)

        for step, imgs in enumerate(bar):
            imgs = imgs.to(DEVICE)
            B    = imgs.size(0)

            # ── Random mask generation ────────────────────────
            n_mask   = int(n_patches * mask_ratio)    # 147 masked
            noise    = torch.rand(B, n_patches, device=DEVICE)
            ids_sort = noise.argsort(dim=1)
            mask     = torch.zeros(B, n_patches,
                                   dtype=torch.bool, device=DEVICE)
            mask.scatter_(1, ids_sort[:, :n_mask], True)  # True = masked

            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=(DEVICE.type == 'cuda')):
                # ── Encode visible patches through ALL ViT blocks ──
                # vis_tokens shape: (B, N_vis = 49, feat_dim)
                vis_tokens = encode_visible(encoder, imgs, mask)

                # ── Decode: reconstruct all patch positions ───
                # pred shape: (B, n_patches, patch_dim)
                pred   = decoder(vis_tokens, mask, n_patches)

                # ── Reconstruction loss on masked patches only ─
                target = patchify(imgs)                    # (B, 196, 768)
                if norm_pix_loss:                          # per-patch normalised targets (He et al. 2022)
                    t_mean = target.mean(dim=-1, keepdim=True)
                    t_var  = target.var(dim=-1, keepdim=True)
                    target = (target - t_mean) / (t_var + 1e-6) ** 0.5

                # Expand mask to match patch_dim for indexing
                # mask_expanded: (B, n_patches, patch_dim)
                mask_expanded = mask.unsqueeze(-1).expand_as(pred)
                loss = ((pred - target) ** 2)[mask_expanded].mean()
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            n_batches  += 1
            bar.set_postfix(loss=f'{loss.item()*accum_steps:.4f}')

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f'  MAE ep {epoch:3d}/{epochs}  '
              f'loss={avg_loss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'({elapsed:.1f} min)')

        # ── Save resume checkpoint every 5 epochs ─────────────
        # At ~5 min/epoch this saves every ~25 min
        # Worst case disconnect loss = 4 epochs (~20 min)
        if epoch % 5 == 0 or epoch == epochs:
            torch.save({
                'epoch'           : epoch,
                'encoder_state'   : encoder.state_dict(),
                'decoder_state'   : decoder.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'scheduler_state' : scheduler.state_dict(),
                'scaler_state'    : scaler.state_dict(),
                'loss_history'    : loss_history,
            }, resume_ckpt)
            print(f'  [CKPT] MAE resume saved (ep {epoch}/{epochs})')

    # ── Save encoder only (discard decoder) ───────────────────
    torch.save({
        'encoder_state': encoder.state_dict(),   # full ViT-S/16 incl. blocks, cls_token, pos_embed, norm
        'feat_dim'     : feat_dim,
        'loss_history' : loss_history,
        'trained_parts': 'patch_embed+pos_embed+cls_token+blocks+norm (decoder discarded)',
        'norm_pix_loss': norm_pix_loss,
    }, final_ckpt)
    print(f'  [DONE] MAE encoder (full ViT-S/16) saved: {final_ckpt}')

    # Delete resume checkpoint — training complete
    if os.path.exists(resume_ckpt):
        os.remove(resume_ckpt)
        print('  Resume checkpoint cleaned up.')

    torch.cuda.empty_cache()
    return final_ckpt


# ── Run MAE ───────────────────────────────────────────────────
vit_encoder_path = pretrain_mae(epochs=150, batch_size=64, accum_steps=2)
VIT_METHOD       = 'mae'


# ── Loss curve ────────────────────────────────────────────────
ckpt_data = torch.load(vit_encoder_path, map_location='cpu')
if 'loss_history' in ckpt_data and ckpt_data['loss_history']:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ckpt_data['loss_history'],
            color='#FF9800', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss (MSE)')
    ax.set_title('MAE Pretraining Loss Curve', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{CONFIG["figures_dir"]}/{VIT_METHOD}_loss_curve.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  Loss curve saved.')

print()
print('=' * 55)
print(f'  CELL 4 COMPLETE — {VIT_METHOD.upper()} done')
print(f'  Encoder : {vit_encoder_path}')
print(f'  feat_dim: {ckpt_data.get("feat_dim", 384)}')
print('=' * 55)