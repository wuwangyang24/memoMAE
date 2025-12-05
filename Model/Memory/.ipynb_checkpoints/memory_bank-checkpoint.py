import torch
import numpy as np
from .recollect_faiss import RecollectFaiss


class MemoryBank:
    """A fixed-size memory bank for storing embeddings."""
    def __init__(self, 
                 capacity: int, 
                 embed_dim: int,
                 normalize: False,
                 device="cuda:0", 
                 dtype=torch.float32) -> None:
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.normalize = normalize
        self.device = device
        self.dtype = dtype
        self.memory, self.scores, self.stored_size = self.reset()
        self.recollector = RecollectFaiss(embed_dim, device=device)

    @torch.no_grad()
    def memorize(
        self,
        items: torch.Tensor,
        scores: torch.Tensor | None = None,
        mode: str = "random",
    ) -> None:
        """
        Add new embeddings to the memory bank.
        Args:
            items: Tensor of shape [B, D]
            scores: Tensor of shape [B] or None
            mode: "random" (replace random items) or "replow" (replace lowest-score items)
        """
        assert mode in {"random", "replow"}, f"Invalid mode: {mode}"

        # --------- Fast device / dtype handling ---------
        # Avoid clone unless we actually need to move / cast
        if items.device != self.device or items.dtype != self.dtype:
            items = items.to(device=self.device, dtype=self.dtype, non_blocking=True)
        else:
            items = items.detach()  # break graph, but no new allocation

        B = items.size(0)
        if B == 0:
            return

        if scores is None:
            # If no scores, just use zeros; replow can't work -> fall back to random
            scores = torch.zeros(B, device=self.device, dtype=self.dtype)
            if mode == "replow":
                mode = "random"
        else:
            if scores.device != self.device or scores.dtype != self.dtype:
                scores = scores.to(device=self.device, dtype=self.dtype, non_blocking=True)
            else:
                scores = scores.detach()

        # --------- Fill free space first ---------
        free = self.capacity - self.stored_size
        if free > 0:
            fill = min(free, B)
            end = self.stored_size + fill
            self.memory[self.stored_size:end].copy_(items[:fill])
            self.scores[self.stored_size:end].copy_(scores[:fill])
            self.stored_size = end
            if fill == B:
                # everything fit into free space
                return
            # Otherwise, we still have overflow to handle
            items = items[fill:]
            scores = scores[fill:]
            B = items.size(0)  # number of overflow items
        # --------- Overflow: memory is full here ---------
        # B > 0 and self.stored_size == self.capacity
        if mode == "random":
            # random replacement
            idx = torch.randint(self.capacity, (B,), device=self.device)
        else:  # "replow"
            # Replace the B lowest-score entries
            # (O(capacity log B); if this is a bottleneck, we can discuss better schemes)
            _, idx = torch.topk(self.scores, k=B, largest=False)
        self.memory[idx].copy_(items)
        self.scores[idx].copy_(scores)

    def recollect(self, query: torch.Tensor, k: int) -> torch.Tensor:
        """Retrieve top-k similar embeddings from memory bank for given queries.
        Args:
            query: Tensor of shape [B, M, D]
            k: number of nearest neighbors to retrieve
        Returns:
            neighbor_embeddings: Tensor of shape [B, M*k, D]
        """
        B, M, D = query.size()
        query = query.reshape(B * M, D)
        if self.stored_size == 0:
            raise ValueError("Memory bank is empty. Cannot perform recollection.")
        # Update FAISS index with current memory
        if self.normalize:
            self.recollector.update_index(self.memory[:self.stored_size]/self.memory[:self.stored_size].norm(dim=1, keepdim=True))
            distances, indices = self.recollector.recollect((query/query.norm(dim=1, keepdim=True)).to(self.device), k+1)  # [B*M, k]
        else:
            self.recollector.update_index(self.memory[:self.stored_size])
            distances, indices = self.recollector.recollect(query.to(self.device), k+1)  # [B*M, k]
        # Drop the first neighbor (assumed to be self)
        indices = indices[:, 1:]
        indices = indices.reshape(B, M * k)
        neighbor_embeddings = self.memory[indices]  # [B, M*k, D]
        return neighbor_embeddings.view(B, M, k, D)

    def reset(self):
        """Reset memory bank (preallocate contiguous memory)."""
        self.memory = torch.empty(
            (self.capacity, self.embed_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.scores = torch.empty(self.capacity, device=self.device, dtype=self.dtype)
        self.stored_size = 0
        return self.memory, self.scores, self.stored_size

    def get_memory(self) -> torch.Tensor:
        return self.memory[:self.stored_size]