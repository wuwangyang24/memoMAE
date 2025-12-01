import timm
import torch.nn as nn
from .asymAttention import AsymAttention
from timm.models.vision_transformer import Attention, Mlp


class AsymBlock(nn.Module):
    '''
    Asymmetric Transformer Block
    '''
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-06)
        self.attn = AsymAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            nn.Identity()
            if drop_path == 0.
            else timm.layers.DropPath(drop_path)
        )
        self.norm2 = norm_layer(dim, eps=1e-06)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None, sim_embeddings=None, return_attn: bool = False):
        '''
        Forward function.
        Args:
            x: input features with shape (B, N, D)
            mask: shape (B, N, 1)
            sim_embeddings: similar embeddings with shape (B, N, K, D)
            return_attn: whether to return attention weights
        Returns:
            output features with shape (B, N, D)
            (optional) attention weights with shape (B, num_heads, N, N+K)
        '''
        if mask is not None:
            mask = mask.view(x.shape[0],-1)
        if sim_embeddings is not None:
            sim_embeddings = self.norm1(sim_embeddings)
        if return_attn:
            out, attn = self.attn(self.norm1(x), mask, sim_embeddings, return_attn=True)
            x = x + self.drop_path(out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask, sim_embeddings, return_attn=False))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        return x
