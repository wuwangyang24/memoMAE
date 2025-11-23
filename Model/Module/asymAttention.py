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
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
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
            sim_embeddings: similar embeddings with shape (B, N, M, D)
            attn_mask: attention mask
        Returns:
            output features with shape (B, N, D)
        '''
        B, N, D = x.shape
        _, _, M, _ = sim_embeddings.shape
        # q: (B, N, D) → (B, #heads, N, head_dim)
        # kv: (B, NK, 2*D) → (2, B, #heads, NK, head_dim)
        # ---- q ----
        q = self.q(x).reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
        # ---- kv x ----
        kv_x = self.kv(x)                          # (B, N, 2D)
        k_x, v_x = kv_x.chunk(2, dim=-1)
        # ---- kv sim ----
        kv_sim = self.kv(sim_embeddings)           # (B, N, M, 2D)
        k_sim, v_sim = kv_sim.chunk(2, dim=-1)
        # concat final K and V
        k = torch.cat([k_x.unsqueeze(1).expand(-1, N, -1, -1), k_sim], dim=2)  # (B, N, N+M, D)
        v = torch.cat([v_x.unsqueeze(1).expand(-1, N, -1, -1), v_sim], dim=2)
        # # reshape for attention
        # k = k.reshape(B, N, -1, self.num_heads, D // self.num_heads).transpose(2, 3)
        # v = v.reshape(B, N, -1, self.num_heads, D // self.num_heads).transpose(2, 3)
        # # edit shapes → attention dims: (B, heads, query_tokens, key_tokens)
        # k = k.reshape(B, self.num_heads, N, -1, D // self.num_heads)
        # v = v.reshape(B, self.num_heads, N, -1, D // self.num_heads)
        L = k.size(2)  # or k.shape[2], should be N+M
        # k, v: (B, N, L, D) -> (B, H, N, L, Hd)
        k = k.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        attn = torch.matmul(q.unsqueeze(3), k.transpose(-2, -1)) * self.scale     # (B, heads, N, N+M)
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
