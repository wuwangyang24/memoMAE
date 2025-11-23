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

    # def forward(self, x, sim_embeddings, attn_mask=None, return_attn: bool = False):
    #     '''
    #     Forward function.
    #     Args:
    #         x: input features with shape (B, N, D)
    #         sim_embeddings: similar embeddings with shape (B, N, M, D)
    #         attn_mask: attention mask
    #     Returns:
    #         output features with shape (B, N, D)
    #     '''
    #     B, N, D = x.shape
    #     _, _, M, _ = sim_embeddings.shape
    #     # q: (B, N, D) → (B, #heads, N, head_dim)
    #     # kv: (B, NK, 2*D) → (2, B, #heads, NK, head_dim)
    #     # ---- q ----
    #     q = self.q(x).reshape(B, N, self.num_heads, D // self.num_heads).transpose(1, 2)
    #     # ---- kv x ----
    #     kv_x = self.kv(x)                          # (B, N, 2D)
    #     k_x, v_x = kv_x.chunk(2, dim=-1)
    #     # ---- kv sim ----
    #     kv_sim = self.kv(sim_embeddings)           # (B, N, M, 2D)
    #     k_sim, v_sim = kv_sim.chunk(2, dim=-1)
    #     # concat final K and V
    #     k = torch.cat([k_x.unsqueeze(1).expand(-1, N, -1, -1), k_sim], dim=2)  # (B, N, N+M, D)
    #     v = torch.cat([v_x.unsqueeze(1).expand(-1, N, -1, -1), v_sim], dim=2)
    #     # # reshape for attention
    #     # k = k.reshape(B, N, -1, self.num_heads, D // self.num_heads).transpose(2, 3)
    #     # v = v.reshape(B, N, -1, self.num_heads, D // self.num_heads).transpose(2, 3)
    #     # # edit shapes → attention dims: (B, heads, query_tokens, key_tokens)
    #     # k = k.reshape(B, self.num_heads, N, -1, D // self.num_heads)
    #     # v = v.reshape(B, self.num_heads, N, -1, D // self.num_heads)
    #     L = k.size(2)  # or k.shape[2], should be N+M
    #     # k, v: (B, N, L, D) -> (B, H, N, L, Hd)
    #     k = k.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    #     v = v.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
    #     attn = torch.matmul(q.unsqueeze(3), k.transpose(-2, -1)) * self.scale     # (B, heads, N, N+M)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
    #     # attention output
    #     x = attn @ v                                      # (B, heads, N, head_dim)
    #     # merge heads: (B, N, D)
    #     x = (
    #         x.transpose(1, 2)
    #          .reshape(B, -1, D)
    #     )
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     if return_attn:
    #         return x, attn
    #     return x

    def forward_base(self, x, attn_mask=None, return_attn: bool = False):
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
        H = self.num_heads
        Dh = self.head_dim
        # 1. Project Q, K, V
        q = self.q(x).view(B, N, H, Dh).permute(0, 2, 1, 3)       # (B, H, N, Dh)
        kv = self.kv(x)                                        # (B, N, 2D)
        k, v = kv.chunk(2, dim=-1)                              # (B, N, D), (B, N, D)
        k = k.view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        v = v.view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        # 2. Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 3. Weighted sum
        out = torch.matmul(attn, v)                             # (B, H, N, Dh)
        # 4. Merge heads
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attn
        return out

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
        kv_x = self.kv(x)                                          # (B, N, 2D)
        k_x, v_x = kv_x.chunk(2, dim=-1)                           # (B, N, D), (B, N, D)
        k_x = k_x.view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        v_x = v_x.view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        # ---- 3. K, V for similar patches (per query) ----
        kv_sim = self.kv(sim_embeddings)                           # (B, N, M, 2D)
        k_sim, v_sim = kv_sim.chunk(2, dim=-1)                     # (B, N, M, D)
        k_sim = k_sim.view(B, N, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, N, M, Dh)
        v_sim = v_sim.view(B, N, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, N, M, Dh)
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
    
    def forward(self, x, sim_embeddings=None, attn_mask=None, return_attn: bool = False):
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
            return self.forward_base(x, attn_mask, return_attn)

