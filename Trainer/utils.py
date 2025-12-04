import torch
import hdbscan
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------------------------------------------------------#
# helper functions
# --------------------------------------------------------------------------------#
def reconstruct_from_pred(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred: (B, L, p*p*3)
    mask: (B, L)  with 1 = masked, 0 = visible (MAE-style)
    returns reconstructed image: (B, 3, H, W)
    """
    if len(pred.shape) == 4:
        return pred
    B, L, _ = pred.shape
    p = self.patch_size
    C = 3
    # (B, L, 3, p, p)
    pred = pred.view(B, L, C, p, p)
    # apply mask: only fill masked patches, keep visible patches as 0 here
    mask_ = mask.view(B, L, 1, 1, 1)                    # (B, L, 1, 1, 1)
    if not torch.all(mask_ == 0).item():
        pred = pred * mask_                        # (B, 3, H, W)
    return pred

def build_masked_image(imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, 3, H, W)
    mask: (B, L)
    returns masked image (visible patches kept, masked -> 0)
    """
    if len(mask.shape) == 4:
        return imgs * (1-mask)
    B, C, H, W = imgs.shape
    p = self.patch_size
    h = H // p
    w = W // p
    # mask: (B, L) → (B, h, w, 1, 1)
    mask_patches = mask.view(B, h, w, 1, 1)
    # upsample mask to pixel space
    mask_img = mask_patches.repeat(1, 1, 1, p, p)            # (B, h, w, p, p)
    mask_img = mask_img.permute(0, 3, 1, 4, 2)               # (B, p, h, p, w)
    mask_img = mask_img.reshape(B, 1, H, W)                  # (B, 1, H, W)
    # visible = 1 - mask
    visible = 1.0 - mask_img
    return imgs * visible
    
def make_reconstruction_images(imgs, masked, rec):
    """
    imgs, masked, rec: (B, 3, H, W)
    """
    n = min(8, imgs.size(0))
    imgs = imgs[:n]
    masked = masked[:n]
    rec = rec[:n]
    # undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
    imgs = imgs * std + mean
    # rec  = rec * std + mean
    # masked = masked * std + mean
    # clamp to [0,1] for visualization
    imgs = imgs.clamp(0, 1)
    masked = masked.clamp(0, 1)
    rec = rec.clamp(0, 1)
    # stack: [orig | masked | recon] horizontally for each sample
    rows = torch.cat([imgs, rec], dim=0)  # (3n, 3, H, W)
    grid = make_grid(rows, nrow=n)                # show n per row: orig/masked/rec in rows
    return grid
    

# def log_attention_maps(batch: torch.Tensor, attn_scores: Optional[torch.Tensor]) -> None:
#     """Log attention maps overlaid on input images to WandB."""
#     # Log first N images and attention maps to WandB
#     N = 8  # number of images to log
#     if attn_scores is not None:
#         images = batch[0]["images"]  # shape [B, C, H, W]
#         batch_size = images.shape[0]
#         N = min(N, batch_size)
#         # CLS token attention to patches
#         cls_attn = attn_scores[:, :, 0, 1:]  # [B, heads, tokens]
#         cls_attn = cls_attn.mean(dim=1)      # average over heads, shape [B, tokens]
#         # Compute patch grid size dynamically
#         patch_size = int(cls_attn.shape[1] ** 0.5)
#         cls_attn_map = cls_attn.reshape(batch_size, patch_size, patch_size)  # [B, H, W]
#         cls_attn_map = torch.nn.functional.interpolate(
#             cls_attn_map.unsqueeze(1),  # add channel dim
#             size=(images.shape[2], images.shape[3]),  # upsample to original size
#             mode='bilinear',
#             align_corners=False
#         )
#         # Normalize attention maps
#         min_vals = torch.amin(cls_attn_map, dim=(1,2,3), keepdim=True)
#         max_vals = torch.amax(cls_attn_map, dim=(1,2,3), keepdim=True)
#         cls_attn_map = (cls_attn_map - min_vals) / (max_vals - min_vals + 1e-8)
#         for i in range(N):
#             img = images[i].cpu()  # [C, H, W]
#             attn = cls_attn_map[i].cpu()  # [1, H, W]
#             # Convert image to [H, W, C] for overlay
#             img_np = img.permute(1, 2, 0).numpy()
#             img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize 0-1
#             attn_np = attn.squeeze(0).numpy()  # [H, W]
#             # Overlay attention on image
#             overlay = 0.5 * img_np + 0.5 * plt.cm.jet(attn_np)[..., :3]
#             overlay = overlay.clip(0, 1)
#             side_by_side = np.concatenate([img_np, overlay, plt.cm.jet(attn_np)[..., :3]], axis=1)
#             # Log single combined image
#             self.logger.experiment.log({
#                 f"val_image_and_attention_{i}": wandb.Image((side_by_side * 255).astype("uint8")),
#                 "epoch": self.current_epoch,
#             })

# def log_attention_distribution(attn: torch.Tensor):
#     """
#     Log the distribution of self and similar attention scores
#     Args:
#         attn: Tensor of shape (B, H, N, N+M)
#         step: Optional step number for W&B
#         prefix: Prefix for W&B plot titles
#     """
#     N, NM = attn.shape[-2], attn.shape[-1]
#     if N == NM:
#         attn_self = attn.numpy().flatten()
#         # Plot histogram using matplotlib
#         plt.figure(figsize=(6, 4))
#         plt.hist(attn_self, bins=50, color='skyblue', alpha=0.7)
#         plt.title(f"Self Attention Distribution")
#         plt.xlabel("Attention score")
#         plt.ylabel("Frequency")
#         fig = plt.gcf()
#     else:
#         # Separate self and similar attention
#         attn_self = attn[..., :N].numpy().flatten()
#         attn_sim  = attn[..., N:].numpy().flatten()
#         # Plot histograms using matplotlib
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#         axes[0].hist(attn_self, bins=50, color='skyblue', alpha=0.7)
#         axes[0].set_title(f"Self Attention Distribution")
#         axes[0].set_xlabel("Attention score")
#         axes[0].set_ylabel("Frequency")
#         axes[1].hist(attn_sim, bins=50, color='salmon', alpha=0.7)
#         axes[1].set_title(f"Similar Patches Attention")
#         axes[1].set_xlabel("Attention score")
#         axes[1].set_ylabel("Frequency")
#         plt.tight_layout()
#     # Log figure to W&B
#     self.logger.experiment.log({" Attention Distribution": wandb.Image(fig), "epoch": self.current_epoch})
#     plt.close(fig)

# def visualize_cluster(memory_embeddings: torch.Tensor):
#     '''Visualize clusters in the memory bank using UMAP and HDBSCAN, and log to WandB.'''
#     memory_embeddings = memory_embeddings.detach().cpu().numpy()
#     n = memory_embeddings.shape[0]
#     # subsample (UMAP + tSNE heavy)
#     if n > 10000:
#         idx = np.random.choice(n, 10000, replace=False)
#         memory_embeddings = memory_embeddings[idx]
#     # ----- Step 1: UMAP (20D) -----
#     umap_reducer = umap.UMAP(
#         n_components=50,
#         n_neighbors=30,
#         min_dist=0.0,
#         metric="cosine",
#         random_state=42,
#     )
#     x_umap = umap_reducer.fit_transform(memory_embeddings)
#     # ----- Step 2: HDBSCAN cluster on 20D UMAP -----
#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=20,
#         metric="euclidean",
#         cluster_selection_method="eom"
#     ).fit(x_umap)
#     labels = clusterer.labels_
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     # ----- Step 3: (optional) t-SNE for plotting only -----
#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         learning_rate="auto",
#         init="pca",
#         random_state=42
#     )
#     x_2d = tsne.fit_transform(x_umap)
#     df = pd.DataFrame({
#         "x": x_2d[:, 0],
#         "y": x_2d[:, 1],
#         "cluster": labels.astype(str),
#     })
#     df["cluster"] = df["cluster"].replace({"-1": "noise"})
#     plt.figure(figsize=(7, 6))
#     sns.scatterplot(
#         data=df,
#         x="x",
#         y="y",
#         hue="cluster",
#         s=10,
#         linewidth=0,
#         legend=False,
#     )
#     plt.title(f"UMAP→HDBSCAN clusters (visualized with t-SNE) — {n_clusters} clusters")
#     plt.tight_layout()
#     self.logger.experiment.log({
#         "cluster_visualization": wandb.Image(plt),
#         "epoch": self.current_epoch,
#         "num_clusters": n_clusters,
#     })
#     plt.close()

def linear_probing(train_feats, train_labels, val_feats, val_labels, batch_size=2048, device='cpu'):
    # ----- 1. Stack features & labels -----
    feats_train = torch.cat(train_feats, dim=0)   # (N, D)
    labels_train = torch.cat(train_labels, dim=0) # (N,)
    feats_test = torch.cat(val_feats, dim=0)   # (N, D)
    labels_test = torch.cat(val_labels, dim=0) # (N,)
    N, D = feats_train.shape
    num_classes = int(labels_train.max().item()) + 1
    # move to device
    if device != 'cpu':
        feats_train = feats_train.to(device)
        labels_train = labels_train.to(device)
        feats_test = feats_test.to(device)
        labels_test = labels_test.to(device)
    # ----- 3. Linear classifier with BatchNorm -----
    clf = nn.Sequential(
        nn.BatchNorm1d(D, affine=False),
        nn.Linear(D, num_classes)
    ).to(device)
    # MAE LR scaling rule
    batch_size = 2048
    batch_size = min(batch_size, feats_train.size(0))
    base_lr = 0.1
    max_lr = base_lr * batch_size / 256.0
    train_loader = DataLoader(
        TensorDataset(feats_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(
        clf.parameters(),
        lr=max_lr,          # final LR, warmup scheduler will scale it
        momentum=0.9,
        weight_decay=0.0,
    )
    # ----- 4. Linear warmup + cosine schedulers -----
    warmup_epochs = 10
    total_epochs = 100
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,       # start at near-zero LR
        end_factor=1.0,          # warm up to max_lr
        total_iters=warmup_epochs,
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()
    # ----- 5. Training -----
    clf.train()
    with torch.enable_grad():
        for epoch in tqdm(range(total_epochs)):
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
    # ----- 6. Evaluation -----
    clf.eval()
    with torch.no_grad():
        preds = clf(feats_test).argmax(dim=1)
        acc = (preds == labels_test).float().mean().item()
    return acc