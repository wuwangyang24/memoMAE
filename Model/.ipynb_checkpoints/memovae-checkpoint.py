import os
from mae import MaskedAutoencoderViT
from Memory.memory_bank import MemoryBank


class memoVAE(MaskedAutoencoderViT):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False,
                 memory_capacity=10000
                ):
        super().__init__(img_size=img_size, 
                         patch_size=patch_size, 
                         in_chans=in_chans,
                         embed_dim=embed_dim, 
                         depth=depth, 
                         num_heads=num_heads,
                         decoder_embed_dim=decoder_embed_dim, 
                         decoder_depth=decoder_depth, 
                         decoder_num_heads=decoder_num_heads,
                         mlp_ratio=mlp_ratio, 
                         norm_layer=norm_layer, 
                         norm_pix_loss=norm_pix_loss)
        self.memory_bank = MemoryBank(
            capacity=memory_capacity,
            embed_dim=embed_dim,
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        )

    def get_k_simpatches(self, k):
        pass

    def forward_memory(self, imgs, memo_ratio=0.5):
        # embed patches
        x = self.patch_embed(imgs)
        # memorize
        # TODO
        return x

    def forward_encoder(self, x, mask_ratio=0.75, k_sim_patches=5):
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] # (B,N,D)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio) # (B,M,D)
        # get similar patches
        x = self.get_k_simpatches(k) # (B,M,k+1,D)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B,M,k+2,D)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward(self, imgs, memo_ratio=0.5, mask_ratio=0.75, k_sim_patches=5):
        x = self.forward_memory(imgs, memo_ratio)
        latents, mask, ids_restore = self.forward_encoder(x, mask_ratio, k_sim_patches)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
        
        
    