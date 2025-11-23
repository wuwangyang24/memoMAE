import timm
import torch.nn as nn
from .asymAttention import AsymAttention 


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
        self.norm1 = norm_layer(dim)
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
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, sim_embeddings, return_attn: bool = False):
        '''
        Forward function.
        Args:
            x: input features with shape (B, N, D)
            sim_embeddings: similar embeddings with shape (B, N, K, D)
            return_attn: whether to return attention weights
        Returns:
            output features with shape (B, N, D)
            (optional) attention weights with shape (B, num_heads, N, N+K)
        '''
        if sim_embeddings is not None:
            sim_embeddings = self.norm1(sim_embeddings)
        if return_attn:
            x, attn = self.attn(self.norm1(x), sim_embeddings, return_attn=True)
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = self.attn(self.norm1(x), sim_embeddings, return_attn=False)
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
