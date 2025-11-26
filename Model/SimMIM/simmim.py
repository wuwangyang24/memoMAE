# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .vision_transformer import VisionTransformer
from ..Memory.memory_bank import MemoryBank
from ..Module.asymBlock import AsymBlock


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        memory_capacity = kwargs.pop("memory_capacity", None) 
        super().__init__(**kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)
        self.memory_bank = MemoryBank(
            capacity=memory_capacity,
            embed_dim=self.embed_dim,
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        )
        self.blocks = nn.ModuleList([
            AsymBlock(self.embed_dim, 
                      self.num_heads, 
                      self.mlp_ratio, 
                      qkv_bias=self.qkv_bias, 
                      norm_layer=nn.LayerNorm)
            for i in range(self.depth)])
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, 
                x, 
                mask, 
                nosim_train, 
                k_sim_patches: int=10, 
                return_attn: bool=False):
        x = self.patch_embed(x)
        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        # push patches to memory bank
        keep = (w.squeeze(-1) == 0)
        self.memory_bank.memorize(x[keep])
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
            return {'z':x, 'attn':attn}
        return {'z':x}


