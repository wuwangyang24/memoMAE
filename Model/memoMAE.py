import os
import torch
import torch.nn as nn
from .MAE.mae import MaskedAutoencoderViT
from .Memory.memory_bank import MemoryBank
from .Module.asymAttention import AsymAttention


class memoMAE(MaskedAutoencoderViT):
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
        # replace standard attn with asymattn
        # for blk in self.blocks:
        #     blk.attn = AsymAttention(
        #                     dim=blk.attn.qkv.in_features,
        #                     num_heads=blk.attn.num_heads,
        #                     qkv_bias=True,
        #                     attn_drop=blk.attn.attn_drop.p if hasattr(blk.attn.attn_drop, "p") else 0.,
        #                     proj_drop=blk.attn.proj_drop.p if hasattr(blk.attn.proj_drop, "p") else 0.,
        #                 )
    
    def forward_encoder(self, x, mask_ratio=0.75, k_sim_patches=5):
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio) # (B, M, D)
        B, N, D = x.shape
        M = x_masked.shape[1]
        ids_keep = torch.nonzero(mask == 0, as_tuple=True)[1].reshape(B, M)
        pos_embed_patches = self.pos_embed[:, 1:, :]  # (1, N, D)
        pos_embed_kept = torch.gather(
            pos_embed_patches.expand(B, -1, -1),
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        sim_patch_embeds = self.memory_bank.recollect(x_masked, k_sim_patches) # (B, M, K, D)
        x_masked = x_masked + pos_embed_kept
        x_masked = torch.cat([x_masked.unsqueeze(2), sim_patch_embeds], dim=2).reshape(B, -1, D) # (B, M*(K+1), D)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]   # (1, 1, D)
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)   # (B, 1+M*(K+1), D)
        for blk in self.blocks:
            x = blk(x)
        x = torch.cat([x[:, :1, :], x[:, 1:, :][:, ::6, :]], dim=1)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward(self, imgs, memo_ratio=0.5, mask_ratio=0.75, k_sim_patches=5):
        x = self.patch_embed(imgs) # (B, N, D)
        self.memory_bank.memorize(x.reshape(-1, x.shape[-1]))
        latents, mask, ids_restore = self.forward_encoder(x, mask_ratio, k_sim_patches) # (B, M, D)
        pred = self.forward_decoder(latents, ids_restore) # (B, N, p*p*3)
        loss = self.forward_loss(imgs, pred, mask)
        return {'loss':loss, 'pred':pred, 'mask': mask}
        
        
    