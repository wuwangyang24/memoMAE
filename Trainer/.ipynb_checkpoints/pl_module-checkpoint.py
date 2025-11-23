import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import lightning as pl
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision.utils import make_grid


class LightningModel(pl.LightningModule):

    def __init__(self, model: Any, config: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.patch_size = config.mae.patch_size
        # Extract optimizer configuration
        optimizer_config = config.optimizer
        self.max_epochs = config.training.max_epochs
        self.lr = optimizer_config.learning_rate
        self.beta1 = optimizer_config.beta1
        self.beta2 = optimizer_config.beta2
        self.weight_decay = optimizer_config.weight_decay
        self.eta_min = optimizer_config.eta_min
        self.warmup_epochs = optimizer_config.warmup_epochs
        self.start_factor = optimizer_config.start_factor
        self.max_epochs = config.training.max_epochs
        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        loss = self.model(batch[0]['images']).get('loss', None)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # --- Log LR every step ---
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        imgs = batch[0]["images"] 
        outputs = self.model(imgs)
        loss = outputs.get('loss', None)
        pred = outputs.get('pred', None)
        mask = outputs.get('mask', None)
        # attn_scores = outputs.get('attn_scores', None)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # only visualise for first batch each epoch
        if batch_idx == 0:
            with torch.no_grad():
                rec = self.reconstruct_from_pred(pred, mask)
                masked = self.build_masked_image(imgs, mask)
                self.log_reconstruction_images(imgs, masked, rec, stage="val")
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if self.model.memory_bank.stored_size > 0:
            self.visualize_cluster(self.model.memory_bank.memory)

    def configure_optimizers(self) -> Dict[str, Any]:
        # ---- compute steps ----
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.max_epochs
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = self.max_epochs * steps_per_epoch
        print(f"Warmup: {warmup_steps} steps, total: {total_steps} steps")
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # ---- schedulers ----
        warmup = LinearLR(
            optimizer,
            start_factor=self.start_factor,
            total_iters=warmup_steps,
        )
    
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=self.eta_min,
        )
    
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    # --------------------------------------------------------------------------------#
    # helper functions
    # --------------------------------------------------------------------------------#
    def reconstruct_from_pred(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, L, p*p*3)
        mask: (B, L)  with 1 = masked, 0 = visible (MAE-style)
        returns reconstructed image: (B, 3, H, W)
        """
        B, L, _ = pred.shape
        p = self.patch_size
        C = 3
        # (B, L, 3, p, p)
        pred = pred.view(B, L, C, p, p)
        # apply mask: only fill masked patches, keep visible patches as 0 here
        mask_ = mask.view(B, L, 1, 1, 1)                    # (B, L, 1, 1, 1)
        pred = pred * mask_
        # unpatchify to full image
        rec = self.model.unpatchify(pred)                         # (B, 3, H, W)
        return rec

    def build_masked_image(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, H, W)
        mask: (B, L)
        returns masked image (visible patches kept, masked -> 0)
        """
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
        
    def log_reconstruction_images(self, imgs, masked, rec, stage="val"):
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
        self.logger.experiment.log({f"{stage}/reconstructions": wandb.Image(grid), "epoch": self.current_epoch})

    def log_attention_maps(self, batch: torch.Tensor, attn_scores: Optional[torch.Tensor]) -> None:
        """Log attention maps overlaid on input images to WandB."""
        # Log first N images and attention maps to WandB
        N = 8  # number of images to log
        if attn_scores is not None:
            images = batch[0]["images"]  # shape [B, C, H, W]
            batch_size = images.shape[0]
            N = min(N, batch_size)
            # CLS token attention to patches
            cls_attn = attn_scores[:, :, 0, 1:]  # [B, heads, tokens]
            cls_attn = cls_attn.mean(dim=1)      # average over heads, shape [B, tokens]
            # Compute patch grid size dynamically
            patch_size = int(cls_attn.shape[1] ** 0.5)
            cls_attn_map = cls_attn.reshape(batch_size, patch_size, patch_size)  # [B, H, W]
            cls_attn_map = torch.nn.functional.interpolate(
                cls_attn_map.unsqueeze(1),  # add channel dim
                size=(images.shape[2], images.shape[3]),  # upsample to original size
                mode='bilinear',
                align_corners=False
            )
            # Normalize attention maps
            min_vals = torch.amin(cls_attn_map, dim=(1,2,3), keepdim=True)
            max_vals = torch.amax(cls_attn_map, dim=(1,2,3), keepdim=True)
            cls_attn_map = (cls_attn_map - min_vals) / (max_vals - min_vals + 1e-8)
            for i in range(N):
                img = images[i].cpu()  # [C, H, W]
                attn = cls_attn_map[i].cpu()  # [1, H, W]
                # Convert image to [H, W, C] for overlay
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize 0-1
                attn_np = attn.squeeze(0).numpy()  # [H, W]
                # Overlay attention on image
                overlay = 0.5 * img_np + 0.5 * plt.cm.jet(attn_np)[..., :3]
                overlay = overlay.clip(0, 1)
                side_by_side = np.concatenate([img_np, overlay, plt.cm.jet(attn_np)[..., :3]], axis=1)
                # Log single combined image
                self.logger.experiment.log({
                    f"val_image_and_attention_{i}": wandb.Image((side_by_side * 255).astype("uint8")),
                    "epoch": self.current_epoch,
                })

    def visualize_cluster(self, memory_embeddings):
        """
        memory_embeddings: numpy array or torch.Tensor of shape (B, D)
        Logs UMAP→tSNE 2D cluster scatterplot to W&B.
        """
        memory_embeddings = memory_embeddings.detach().cpu().numpy()
        n = memory_embeddings.shape[0]
        idx = np.random.choice(n, 5000, replace=False)
        memory_embeddings = memory_embeddings[idx]
        umap_reducer = umap.UMAP(
            n_components=20,
            n_neighbors=30,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        x_umap = umap_reducer.fit_transform(memory_embeddings)
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42
        )
        x_2d = tsne.fit_transform(x_umap)
        db = DBSCAN(eps=0.5, min_samples=10).fit(x_2d)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        df = pd.DataFrame({
            "x": x_2d[:, 0],
            "y": x_2d[:, 1],
            "cluster": labels.astype(str)
        })
        df["cluster"] = df["cluster"].replace({"-1": "noise"})
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="cluster",
            s=10,
            linewidth=0,
            legend=False
        )
        plt.title(f"t-SNE Cluster Visualization — {n_clusters} clusters")
        plt.tight_layout()
        self.logger.experiment.log({
            "cluster_visualization": wandb.Image(plt),
            "epoch": self.current_epoch,
            "num_clusters": n_clusters
        })
        plt.close()