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
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward_base(self, x, return_attn: bool = False):
        """
        Standard multi-head self-attention forward function.
        Args:
            x: input features with shape (B, N, D)
            attn_mask: optional attention mask (B, N, N)
            return_attn: if True, also return attention weights
        Returns:
            output features with shape (B, N, D)
        """
        B, N, D = x.shape
        # 1. Project Q, K, V
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # (B, H, N, Dh)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # (B, H, N, Dh)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # (B, H, N, Dh)
        # scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)        # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # weighted sum of values
        x = attn @ v                            # (B, H, N, head_dim)
        # merge heads: (B, H, N, head_dim) -> (B, N, dim)
        x = x.transpose(1, 2).reshape(B, N, D)
        # final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        return x

    def forward_asym(self, x, sim_embeddings, return_attn: bool = False):
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
        H = self.num_heads
        Dh = self.head_dim
        # ---- 1. Project Q ----
        q = self.q(x).view(B, N, H, Dh).permute(0, 2, 1, 3)        # (B, H, N, Dh)
        # ---- 2. K, V for self patches ----
        k_x = self.k(x).view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        v_x = self.v(x).view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        # ---- 3. K, V for similar patches (per query) ----
        k_sim = self.k(sim_embeddings).view(B, N, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, N, M, Dh)
        v_sim = self.v(sim_embeddings).view(B, N, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, N, M, Dh)
        # ---- 4. Attention logits ----
        # self-part: (B, H, N, Dh) x (B, H, Dh, N) → (B, H, N, N)
        logits_self = torch.matmul(q, k_x.transpose(-2, -1)) * self.scale
        # sim-part (per n): inner product of q[..., n, :] with each of its M neighbors
        # q:     (B, H, N, 1, Dh)
        # k_sim: (B, H, N, M, Dh)
        # → (B, H, N, M)
        logits_sim = (q.unsqueeze(3) * k_sim).sum(dim=-1) * self.scale  
        # Combine: (B, H, N, N+M)
        logits = torch.cat([logits_self, logits_sim], dim=-1)
        # ---- 5. Softmax over all N+M keys ----
        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)                                # (B, H, N, N+M)
        # Split back
        attn_self = attn[..., :N]                                  # (B, H, N, N)
        attn_sim  = attn[..., N:]                                  # (B, H, N, M)
        # ---- 6. Weighted sum for values ----
        # self values: (B, H, N, Dh), attn_self: (B, H, N, N)
        out_self = torch.matmul(attn_self, v_x)                    # (B, H, N, Dh)
        # sim values: per n, weights over its M neighbors
        # attn_sim: (B, H, N, M) → (B, H, N, M, 1)
        out_sim = (attn_sim.unsqueeze(-1) * v_sim).sum(dim=3)      # (B, H, N, Dh)
        # total output per query
        out = out_self + out_sim                                   # (B, H, N, Dh)
        # merge heads back to (B, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attn
        return out
    
    def forward(self, x, sim_embeddings=None, return_attn: bool = False):
        '''
        Forward function that selects between asymmetric and standard attention.
        Args:
            x: input features with shape (B, N, D)
            sim_embeddings: similar embeddings with shape (B, N, M, D) or None
            attn_mask: attention mask
            return_attn: if True, also return attention weights
        Returns:
            output features with shape (B, N, D)
        '''
        if sim_embeddings is not None:
            return self.forward_asym(x, sim_embeddings, return_attn)
        else:
            return self.forward_base(x, return_attn)

