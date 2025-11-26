import os
import torch
import torch.nn as nn
from .MAE.mae import MaskedAutoencoderViT
from .Memory.memory_bank import MemoryBank
from .Module.asymBlock import AsymBlock


class memoMAE(MaskedAutoencoderViT):
    '''
    Memoized Masked Autoencoder with Asymmetric Attention
    '''
    def __init__(self, config):
        super().__init__(img_size=config.mae.img_size,
                         patch_size=config.mae.patch_size,
                         in_chans=config.mae.in_chans,
                         embed_dim=config.mae.embed_dim,
                         depth=config.mae.depth,
                         num_heads=config.mae.num_heads,
                         decoder_embed_dim=config.mae.decoder_embed_dim,
                         decoder_depth=config.mae.decoder_depth,
                         decoder_num_heads=config.mae.decoder_num_heads,
                         mlp_ratio=config.mae.mlp_ratio,
                         norm_layer=nn.LayerNorm,
                         norm_pix_loss=config.mae.norm_pix_loss)
        self.memory_bank = MemoryBank(
            capacity=config.memory_bank.memory_capacity,
            embed_dim=config.mae.embed_dim,
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        )
        self.blocks = nn.ModuleList([
            AsymBlock(config.mae.embed_dim, 
                      config.mae.num_heads, 
                      config.mae.mlp_ratio, 
                      qkv_bias=config.mae.qkv_bias, 
                      norm_layer=nn.LayerNorm)
            for i in range(config.mae.depth)])
        self.initialize_weights()
    
    def forward_encoder_memo(self, x, mask_ratio=0.75, k_sim_patches=5, return_attn: bool=False):
        '''
        Forward function of encoder.
        Args:
            x: input features with shape (B, N, D)
            mask_ratio: masking ratio
            k_sim_patches: number of similar patches to retrieve from memory bank
        Returns:
            latent features with shape (B, M, D)
            mask: mask indicating which patches are masked
            ids_restore: indices to restore original ordering
        '''
        # patch encoding images with positional embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        # push patches to memory bank
        self.memory_bank.memorize(x.reshape(-1, x.shape[-1]))
        # random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # (B, M, D)
        # retrieve nearest neighbors
        if k_sim_patches > 0:
            sim_patch_embeds = self.memory_bank.recollect(x, k_sim_patches) # (B, M, K, D)
        else:
            sim_patch_embeds = None  # None when no similar patches
        attn = None
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x, sim_patch_embeds, True)
            else:
                x = blk(x, sim_patch_embeds, False)
        x = self.norm(x)
        if attn is not None:
            return x, mask, ids_restore, attn
        return x, mask, ids_restore

    def forward(self, 
                imgs, 
                mask_ratio=0.75, 
                nosim_train: bool=False, 
                num_sim_patches: int=5, 
                return_attn: bool=False,
                return_latents: bool=False
               ):
        '''
        Forward function.
        Args:
            imgs: input images with shape (B, 3, H, W)
            memo_ratio: ratio of patches to memorize
            mask_ratio: masking ratio
            k_sim_patches: number of similar patches to retrieve from memory bank
        Returns:
            loss: reconstruction loss
            pred: predicted pixel values for masked patches
            mask: mask indicating which patches are masked
        '''
        if nosim_train:
            num_sim_patches = 0
        attn = None
        if return_attn:
            latents, mask, ids_restore, attn = self.forward_encoder_memo(imgs, mask_ratio, num_sim_patches, True)
        else:
            latents, mask, ids_restore = self.forward_encoder_memo(imgs, mask_ratio, num_sim_patches)
        pred = self.forward_decoder(latents, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        if return_latents:
            return {'loss': loss, 'pred': pred, 'mask': mask, 'attn': attn, 'latents': latents}
        return {'loss': loss, 'pred': pred, 'mask': mask, 'attn': attn}
        
        
    