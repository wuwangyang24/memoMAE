import torch
import torch.nn as nn
import xformers.ops as xops


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
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def _expand(self, mask, inputs, fill_value=0.0):
        """
        Expand inputs from (B, H, L, Dh) back to (B, H, N, Dh)
        using a boolean mask: 0 = keep, 1 = masked.
    
        fill_value: value for masked positions (0, -inf, etc.)
        """
        # mask: (B, N, 1)
        B, H, L, Dh = inputs.shape
        _, N = mask.shape
        # True where we KEEP (mask == 0)
        keep_bool = (mask == 0)   # (B, N)
        # Number of kept tokens per batch (assumed constant across batch)
        L = keep_bool.sum(dim=1)[0].item()
        # Build idx_keep (B, L)
        _, pos_idx = keep_bool.nonzero(as_tuple=True)
        idx_keep = pos_idx.view(B, L)
        # Create full output tensor, filled with fill_value
        outputs = torch.full(
            (B, H, N, Dh),
            fill_value,
            device=inputs.device,
            dtype=inputs.dtype
        )
        # Expand idx_keep to match scatter shape
        idx_expanded = idx_keep.unsqueeze(1).unsqueeze(-1).expand(B, H, L, Dh)
        # Scatter inputs into correct N positions
        outputs.scatter_(-2, idx_expanded, inputs)
        return outputs


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
            return x, None, attn
        return x, None

    # def forward_partial_asym(self, x, mask, sim_embeddings, return_attn: bool = False):  
    #     """
    #     Partial multiheaded attention
    #     Args:
    #         x: input features with shape (B, N, D)
    #         sim_embeddings: similar embeddings with shape (B, N, M, D)
    #         return_attn: if True, would return attention weights (NOT supported here)
    #     Returns:
    #         output_x: (B, N, D)
    #         output_sim: (B, N, M, D)
    #     """
    #     B, N, D = x.shape
    #     _, _, M, _ = sim_embeddings.shape
    #     L = M + N
    #     x_sim = torch.cat((x.unsqueeze(2).expand(B, N, N, D), sim_embeddings), dim=2) # (B, N, M+N, D)
    
    #     # Project Q, K, V
    #     # q, k ,v: (B*N, N+M, H, Dh)
    #     q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, Dh)
    #     k = self.k(x_sim).reshape(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4) # (B, H, N, L, Dh)
    #     v = self.v(x_sim).reshape(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4) # (B, H, N, L, Dh)

    #     scores = torch.einsum('bhnd,bhnld->bhnl', q, k) * self.scale
    #     scores = torch.softmax(scores, dim=-1) # (B, H, N, L)
    #     out = torch.einsum('bhnl,bhnld->bhnd', scores, v)  # (B, H, N, Dh)
    #     out = out.permute(0, 2, 1, 3).reshape(B, N, D)
    #     sim_out = v
    #     return out, sim_out

    def forward_partial(self, x, mask, sim_embeddings, return_attn: bool = False):  
        """
        Partial multiheaded attention
        Args:
            x: input features with shape (B, N, D)
            sim_embeddings: similar embeddings with shape (B, N, M, D)
            return_attn: if True, would return attention weights (NOT supported here)
        Returns:
            output_x: (B, N, D)
            output_sim: (B, N, M, D)
        """
        B, N, D = x.shape
        _, _, M, _ = sim_embeddings.shape
        BN = B*N
        MN = M + N
        x = x.unsqueeze(1).expand(B, N, N, D)
        x = torch.cat((x, sim_embeddings), dim=2)
        x = x.reshape(BN, MN, D)
        # ---- Project Q, K, V ---- #
        # q, k ,v: (B*N, N+M, H, Dh)
        q = self.q(x).reshape(BN, MN, self.num_heads, self.head_dim)
        k = self.k(x).reshape(BN, MN, self.num_heads, self.head_dim)
        v = self.v(x).reshape(BN, MN, self.num_heads, self.head_dim)
        
        # xFormers memory-efficient attention:
        # output shape: (B*N, N+M, H, Dh)
        # We can pass scale instead of manually scaling q
        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            p=0.,
            scale=self.scale,
        )
        # ---- Merge heads: (B*N, N+M, H, Dh) -> (B, N, N+M, D) ---- #
        out = out.reshape(BN, MN, D)
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        # ---- Split back into x and sim_embeddings ---- #
        out = out.reshape(B, N, MN, D)
        idx = torch.arange(N, device=x.device) 
        x_out = out[:, idx, idx] # (B, N, D)
        sim_out = out[:, :, N:]  # (B, N, M, D)
        return x_out, sim_out

    # def forward_full(self, x, mask, sim_embeddings, return_attn: bool = False):
    #     """
    #     Multi-head self-attention forward function using xFormers memory_efficient_attention.
    #     Args:
    #         x: input features with shape (B, N, D)
    #         sim_embeddings: similar embeddings with shape (B, N, M, D)
    #         return_attn: if True, would return attention weights (NOT supported here)
    #     Returns:
    #         output_x: (B, N, D)
    #         output_sim: (B, N, M, D)
    #     """
    #     if return_attn:
    #         raise NotImplementedError(
    #             "xFormers memory_efficient_attention does not return the full "
    #             "attention matrix. Use the standard attention path if you need attn."
    #         )
    #     B, N, D = x.shape
    #     _, _, M, _ = sim_embeddings.shape
    #     NM = N + N * M

    #     # Concatenate x and sim_embeddings along sequence dimension
    #     # x_cat: (B, NM, D)
    #     x_cat = torch.cat([x, sim_embeddings.reshape(B, N * M, D)], dim=1)
    
    #     # Project Q, K, V
    #     # q, k, v: (B, NM, H, Dh)
    #     q = self.q(x_cat).reshape(B, NM, self.num_heads, self.head_dim)
    #     k = self.k(x_cat).reshape(B, NM, self.num_heads, self.head_dim)
    #     v = self.v(x_cat).reshape(B, NM, self.num_heads, self.head_dim)
    
    #     # xFormers memory-efficient attention:
    #     # output shape: (B, NM, H, Dh)
    #     # We can pass scale instead of manually scaling q
    #     attn_dropout_p = self.attn_drop.p if self.training else 0.0
    #     out = xops.memory_efficient_attention(
    #         q, k, v,
    #         attn_bias=None,
    #         p=attn_dropout_p,
    #         scale=self.scale,  # typically 1/sqrt(Dh)
    #     )
    
    #     # Merge heads: (B, NM, H, Dh) -> (B, NM, D)
    #     out = out.reshape(B, NM, D)
    
    #     # Final projection
    #     out = self.proj(out)
    #     out = self.proj_drop(out)
    
    #     # Split back into x and sim_embeddings
    #     sim_out = out[:, N:, :].reshape(B, N, M, D)
    #     x_out = out[:, :N, :]
    
    #     return x_out, sim_out


    def forward_asym(self, x, mask, sim_embeddings, return_attn: bool = False):
        '''
        Forward function.
        Args:
            x: input features with shape (B, N, D)
            sim_embeddings: similar embeddings with shape (B, L, M, D)
            attn_mask: attention mask
        Returns:
            output features with shape (B, N, D)
        '''
        B, N, D = x.shape
        _, L, M, _ = sim_embeddings.shape
        H = self.num_heads
        Dh = self.head_dim
        reindex_attn = N > L

        # ---- 1. Project Q ----
        q = self.q(x).view(B, N, H, Dh).permute(0, 2, 1, 3)  #(B, H, N, Dh)
        # if masked patches are also passed in, remove them for q_sim
        if reindex_attn:
            mask_bool = (1-mask).bool()
            mask_flat = mask_bool.unsqueeze(1).expand(B, H, N).reshape(B * H, N)
            q_sim = q.reshape(B * H, N, Dh)[mask_flat].reshape(B, H, -1, Dh)  #(B, H, L, Dh)
        else:
            q_sim = None

        # ---- 2. K, V for self patches ----
        k_x = self.k(x).view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        v_x = self.v(x).view(B, N, H, Dh).permute(0, 2, 1, 3)            # (B, H, N, Dh)
        
        # ---- 3. K, V for similar patches (per query) ----
        k_sim = self.k(sim_embeddings).view(B, L, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, L, M, Dh)
        v_sim = self.v(sim_embeddings).view(B, L, M, H, Dh).permute(0, 3, 1, 2, 4)  # (B, H, L, M, Dh)  
        
        # ---- 4. Attention logits ----
        # SELF-part: (B, H, N, Dh) x (B, H, Dh, N) → (B, H, N, N)
        logits_self = torch.matmul(q, k_x.transpose(-2, -1)) * self.scale
        
        # SIM-part: (B, H, L, 1, Dh) x (B, H, L, M, Dh) → (B, H, N, M)
        if q_sim is not None:
            logits_sim = (q_sim.unsqueeze(3) * k_sim).sum(dim=-1) * self.scale  # (B, H, L, M)
            logits_sim = self._expand(mask, logits_sim, fill_value=float('-inf'))  # (B, H, N, M)
        else:
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
        # SELF values: (B, H, N, Dh), attn_self: (B, H, N, N)
        out_self = torch.matmul(attn_self, v_x)                    # (B, H, N, Dh)
        # SIM values: per n, weights over its M neighbors
        # attn_sim: (B, H, L, M) → (B, H, L, M, 1)
        # if N != L, select L from N
        if reindex_attn:
            attn_sim = attn_sim.view(B * H, N, M)[mask_flat].reshape(B, H, -1, M)  #(B, H, L, M)
            out_sim = (attn_sim.unsqueeze(-1) * v_sim).sum(dim=3)  #(B, H, L, Dh)
            out_sim = self._expand(mask, out_sim, 0.)  #(B, H, L, Dh) --> (B, H, N, Dh)
        else:
            out_sim = (attn_sim.unsqueeze(-1) * v_sim).sum(dim=3)  # (B, H, N, Dh)

        # total output per query
        out = out_self + out_sim  # (B, H, N, Dh)
        # merge heads back to (B, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, v_sim.permute(0, 2, 3, 1, 4).reshape(B, L, M, D), attn
        return out, v_sim.permute(0, 2, 3, 1, 4).reshape(B, L, M, D)
    
    def forward(self, x, mask=None, sim_embeddings=None, return_attn: bool = False):
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
            return self.forward_partial(x, mask, sim_embeddings, return_attn)
        else:
            return self.forward_base(x, return_attn)

