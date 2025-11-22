import torch
import torch.nn as nn


class AsymAttention(nn.Module):
    '''
    Asymmetric Multi-Head Self-Attention Module
    '''
    def __init__(
            self,
            dim=768,
            num_heads=8,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # q and kv projection
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, sim_embeddings, attn_mask=None, return_attn: bool = False):
        '''
        Forward function.
        Args:
            x: input features with shape (B, N, D)
            sim_embeddings: similar embeddings with shape (B, N, K, D)
            attn_mask: attention mask
        Returns:
            output features with shape (B, N, D)
        '''
        B, N, D = x.shape
        B, _, K, _ = sim_embeddings.shape
        # q: (B, N, D) → (B, #heads, N, head_dim)
        # kv: (B, NK, 2*D) → (2, B, #heads, NK, head_dim)
        q = (
            self.q(torch.cat([x[:, :1, :], x[:, 1:, :][:, ::6, :]], dim=1))
            .reshape(B, -1, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x.view(B, NK, D))
            .reshape(B, NK, 2, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attention output
        x = attn @ v                                      # (B, heads, N, head_dim)
        # merge heads: (B, N, D)
        x = (
            x.transpose(1, 2)
             .reshape(B, -1, D)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x
