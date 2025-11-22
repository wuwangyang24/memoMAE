import torch.nn as nn


class AsymAttention(nn.Module):
    def __init__(
            self,
            dim=768,
            num_heads=8,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
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

    def forward(self, x):
        B, N, K, D = x.shape # K = 1 + number of similar patches
        NK = N*K

        # q: (B, N, D) → (B, #heads, N, head_dim)
        # kv: (B, NK, 2*D) → (2, B, #heads, NK, head_dim)
        q = (
            self.q(x[:, :, 0, :])
            .reshape(B, N, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x.view(B, NK, D))
            .reshape(B, NK, 2, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[1], kv[2]

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale     # (B, heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        x = attn @ v                                      # (B, heads, N, head_dim)

        # merge heads: (B, N, D)
        x = (
            x.transpose(1, 2)
             .reshape(B, N, D)
        )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
