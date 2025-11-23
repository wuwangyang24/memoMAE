import os
import torch
import torch.nn as nn
from .MAE.mae import MaskedAutoencoderViT
from .Memory.memory_bank import MemoryBank
from .Module.asymBlock import AsymBlock
from .Module.asymAttention import AsymAttention


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
        # replace blocks
        self.blocks = nn.ModuleList([
            AsymBlock(config.mae.embed_dim, 
                      config.mae.num_heads, 
                      config.mae.mlp_ratio, 
                      qkv_bias=config.mae.qkv_bias, 
                      norm_layer=nn.LayerNorm)
            for i in range(config.mae.depth)])
    
    def forward_encoder(self, x, mask_ratio=0.75, k_sim_patches=5):
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
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio) # (B, M, D)
        B, N, D = x.shape
        M = x_masked.shape[1]
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1].reshape(B, M)
        pos_embed_kept = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        sim_patch_embeds = self.memory_bank.recollect(x_masked, k_sim_patches) # (B, M, K, D)
        x_masked = x_masked + pos_embed_kept
        for blk in self.blocks:
            x_masked = blk(x_masked, sim_patch_embeds)
        x_masked = self.norm(x_masked)
        return x_masked, mask, ids_restore

    def forward(self, imgs, memo_ratio=0.5, mask_ratio=0.75, num_sim_patches=5):
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
        x = self.patch_embed(imgs) # (B, N, D)
        self.memory_bank.memorize(x.reshape(-1, x.shape[-1]))
        latents, mask, ids_restore = self.forward_encoder(x, mask_ratio, num_sim_patches) # (B, M, D)
        pred = self.forward_decoder(latents, ids_restore) # (B, N, p*p*3)
        loss = self.forward_loss(imgs, pred, mask)
        return {'loss':loss, 'pred':pred, 'mask': mask}
        
        
    