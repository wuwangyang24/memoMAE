import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .SimMIM.mask_generator import MaskGenerator
from .SimMIM.simmim import VisionTransformerForSimMIM


class MemoSimMIM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder, self.encoder_stride = build_simmim(config)
        self.mask_generator = MaskGenerator(
            input_size=config.data.img_size,
            mask_patch_size=config.vit.patch_size,
            model_patch_size=config.vit.patch_size
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, 
                x, 
                mask_ratio: float=0.6, 
                nosim_train: bool=False, 
                num_sim_patches: int=10, 
                return_attn: bool=False, 
                return_latents: bool=False):
        mask = self.mask_generator(batch_size=x.shape[0], mask_ratio=mask_ratio).to(x.device)
        outputs = self.encoder(x, mask, nosim_train, num_sim_patches)
        z = outputs.get('z', None)
        attn = outputs.get('attn', None)
        # reshape z for decoding
        B, L, C = z.shape
        H = W = int(L ** 0.5)
        z = z.permute(0, 2, 1).reshape(B, C, H, W)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        if return_latents:
            return {'loss': loss, 'pred': x_rec, 'mask': mask, 'attn': attn, 'latents': outputs.get('z', None)}
        return {'loss': loss, 'pred': x_rec, 'mask': mask, 'attn': attn}

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    model_type = 'vit'
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.data.img_size,
            patch_size=config.vit.patch_size,
            in_chans=config.vit.in_chans,
            num_classes=0,
            embed_dim=config.vit.embed_dim,
            depth=config.vit.depth,
            num_heads=config.vit.num_heads,
            mlp_ratio=config.vit.mlp_ratio,
            qkv_bias=config.vit.qkv_bias,
            drop_rate=config.vit.drop_rate,
            drop_path_rate=config.vit.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.vit.init_values,
            use_abs_pos_emb=config.vit.use_ape,
            use_rel_pos_bias=config.vit.use_rpb,
            use_shared_rel_pos_bias=config.vit.use_shared_rpb,
            use_mean_pooling=config.vit.use_mean_pooling,
            memory_capacity=config.memory_bank.memory_capacity
        )
        encoder_stride = config.vit.encoder_stride
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")
    return encoder, encoder_stride