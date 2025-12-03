import torch
import faiss
import faiss.contrib.torch_utils as faiss_torch


class RecollectFaiss:
    """FAISS-based kNN searcher (inner product, GPU optional)."""

    def __init__(self, embed_dim: int, device: str = "cuda:0", use_fp16: bool = False):
        """
        Args:
            embed_dim: embedding dimension D
            device: "cpu" or "cuda:<id>"
            use_fp16: if True and on GPU, use FP16 for index storage/computation
        """
        self.embed_dim = embed_dim
        self.use_fp16 = use_fp16

        if device == "cpu":
            self.device_type = "cpu"
            self.device_id = None
            self.res = None
            self.index = faiss.IndexFlatIP(embed_dim)
        else:
            # parse device like "cuda:0"
            if device.startswith("cuda"):
                self.device_type = "gpu"
                self.device_id = int(device.split(":")[1]) if ":" in device else 0
            else:
                raise ValueError(f"Unsupported device: {device}")
            self.res = faiss.StandardGpuResources()
            # Config for GPU index
            cfg = faiss.GpuIndexFlatConfig()
            cfg.device = self.device_id
            cfg.useFloat16 = use_fp16
            # Direct GPU index (no CPU->GPU copy every time)
            self.index = faiss.GpuIndexFlatIP(self.res, embed_dim, cfg)

    def _prepare_tensor(self, x: torch.Tensor, for_index: bool) -> torch.Tensor:
        """
        Make sure x is contiguous float32 and on the right device
        for FAISS (CPU or GPU).
        """
        # detach and contiguous
        x = x.detach()
        if not x.is_contiguous():
            x = x.contiguous()
        # dtype: FAISS torch utils want float32, but GpuIndexFlat can
        # internally store in float16 if cfg.useFloat16=True.
        if x.dtype != torch.float32:
            x = x.float()
        if self.device_type == "cpu":
            if x.is_cuda:
                x = x.cpu()
        else:  # GPU index
            if not x.is_cuda:
                x = x.to(f"cuda:{self.device_id}")
        return x

    def update_index(self, memory: torch.Tensor) -> None:
        """
        memory: (M, D) tensor
        Rebuilds the index with the current memory.
        """
        if memory.numel() == 0:
            return
        mem = self._prepare_tensor(memory, for_index=True)
        self.index.reset()
        # thanks to faiss.contrib.torch_utils you can pass a torch tensor directly
        self.index.add(mem)

    def recollect(self, query: torch.Tensor, k: int):
        """
        query: (N, D)
        Returns:
            distances: (N, k)
            indices:   (N, k)
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        q = self._prepare_tensor(query, for_index=False)
        D, I = self.index.search(q, k)
        return D, I