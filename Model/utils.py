from memoMAE import memoMAE

def build_model(config):
    model = memoMAE(
        img_size=config.mae.img_size, 
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
        norm_pix_loss=config.mae.norm_pix_loss,
        memory_capacity=config.memory_bank.memory_capacity
    )