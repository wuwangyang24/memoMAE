import torch
import wandb
import lightning as pl
from tqdm import tqdm
from typing import Any, Dict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from .utils import reconstruct_from_pred, build_masked_image, make_reconstruction_images, linear_probing


class LightningModel(pl.LightningModule):

    def __init__(self, model: Any, config: Any) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # hyperparameters
        self.patch_size = config.data.patch_size
        self.nosim_train_epochs = config.hyperparameters.nosim_train_epochs
        self.return_attn = config.vit.return_attn
        self.mask_ratio = config.vit.mask_ratio
        self.num_sim_patches = config.hyperparameters.num_neighbors
        self.lp_device = config.linearprobing.device
        self.baseline = config.hyperparameters.nosim_train_epochs == config.training.max_epochs #if training a baseline
        
        # Extract optimizer configuration
        optimizer_config = config.optimizer
        self.max_epochs = config.training.max_epochs
        self.eval_every = config.training.val_every_epochs
        self.lr = optimizer_config.learning_rate
        self.beta1 = optimizer_config.beta1
        self.beta2 = optimizer_config.beta2
        self.weight_decay = optimizer_config.weight_decay
        self.eta_min = optimizer_config.eta_min
        self.warmup_epochs = optimizer_config.warmup_epochs
        self.start_factor = optimizer_config.start_factor
        self.max_epochs = config.training.max_epochs
        # attention storage for logging
        self.accu_attn_scores = []
        self.feats_val = []
        self.labels_val = []
        self.feats_train = []
        self.labels_train = []

        
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Any:
        nosim_train = self.current_epoch < self.nosim_train_epochs #decide whether to disable similar patches during training
        loss = self.model(batch[0]['images'],  
                          mask_ratio=self.mask_ratio, 
                          num_sim_patches=self.num_sim_patches,
                          memorize=not self.baseline, #during training, memory bank is updated for non baseline; frozen if baseline
                          nosim_train=nosim_train, 
                          return_attn=self.return_attn).get('loss', None)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # --- Log LR every step ---
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        nosim_train = self.current_epoch < self.nosim_train_epochs # decide whether to disable similar patches during validation
        imgs = batch[0]["images"] 
        labels = batch[0]['labels']
        return_attn = self.return_attn and (batch_idx % 10 == 0) #attn scores are stored at certain interval
        outputs = self.model(imgs, 
                             mask_ratio=0, #all patches are visible during validation
                             num_sim_patches=self.num_sim_patches,
                             nosim_train=nosim_train, 
                             memorize=False, #during validation, memory bank is frozen
                             return_attn=return_attn, 
                             return_latents=True)
        imgs = imgs.detach().cpu()
        labels = labels.detach().cpu()
        latents = outputs.get('latents', None).detach().cpu()
        pred = outputs.get('pred', None).detach().cpu()
        mask = outputs.get('mask', None).detach().cpu()
        attn_scores = outputs.get('attn', None)
        # store attention scores for logging
        if attn_scores is not None:
            self.accu_attn_scores.append(attn_scores.detach().cpu())
        # only visualise for first batch each epoch
        if batch_idx == 0:
            with torch.no_grad():
                # log reconstruction images
                rec = reconstruct_from_pred(pred, mask)
                # unpatchify to full image
                rec = self.model.unpatchify(rec) 
                masked = build_masked_image(imgs, mask)
                grid = make_reconstruction_images(imgs, masked, rec)
                self.logger.experiment.log({f"{stage}/reconstructions": wandb.Image(grid), "epoch": self.current_epoch})
        # store latents for linear probing
        self.feats_val.append(latents.mean(1))
        self.labels_val.append(labels.squeeze(1).to(torch.long))

    def on_train_epoch_end(self):
        # Only compute before validation epoch
        if (self.current_epoch + 1) % self.eval_every != 0:
            return
        self.print(f"Preparing latents for training linear prober at epoch {self.current_epoch}...")
        train_loader = self.trainer.datamodule.train_dataloader()
        with torch.no_grad():
            for batch in tqdm(train_loader):
                x = batch[0]["images"] 
                y = batch[0]['labels']
                feats = self.model.forward_encoder_memo(x, 
                                                        mask_ratio=0, #all patches are visible
                                                        k_sim_patches=self.num_sim_patches,
                                                        memorize=False #memory bank is frozen
                                                       )[0]
                self.feats_train.append(feats.mean(1).cpu())
                self.labels_train.append(y.squeeze(1).to(torch.long).cpu())
       
    def on_validation_epoch_end(self):
        # if hasattr(self.model, "memory_bank"):
        #     mb = self.model.memory_bank
        # elif hasattr(self.model.encoder, "memory_bank"):
        #     mb = self.model.encoder.memory_bank
        # else:
        #     mb = None
        # # if no memory bank found
        # if mb is None:
        #     return
        # proceed only if it has something stored
        # if hasattr(mb, "stored_size") and mb.stored_size > 0:
        #     self.visualize_cluster(mb.memory)
        # # log attention distribution
        # if len(self.accu_attn_scores) > 0:
        #     attn_scores = torch.cat(self.accu_attn_scores, dim=0)
        #     self.log_attention_distribution(attn_scores)
        #     self.accu_attn_scores = []  # reset for next epoch
        # ----- linear probing ------ 
        acc = linear_probing(train_feats=self.feats_train, 
                             train_labels=self.labels_train,
                             val_feats=self.feats_val,
                             val_labels=self.labels_val,
                             batch_size=2048,
                             device=self.lp_device)
        self.log("linear_probing_acc", acc, prog_bar=True)
        # reset buffers
        self.feats_val = []
        self.labels_val = []
        self.feats_train = []
        self.labels_train = []

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
    # Save and load memory
    # --------------------------------------------------------------------------------#
    # def on_save_checkpoint(self, checkpoint):
    #     # Save memory tensor
    #     checkpoint["memory_bank"] = self.model.memory_bank.memory.cpu()

    # def on_load_checkpoint(self, checkpoint):
    #     if "memory_bank" in checkpoint:
    #         self.model.memory_bank.memory = checkpoint["memory_bank"].to(self.device)


