import numpy as np
import torch

class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=32):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        
    def __call__(self, mask_ratio, batch_size):
        mask_count = int(np.ceil(self.token_count * mask_ratio))
        # (B, token_count) — initialise all zeros
        mask = np.zeros((batch_size, self.token_count), dtype=int)

        # generate mask for each sample
        for b in range(batch_size):
            idx = np.random.permutation(self.token_count)[:mask_count]
            mask[b, idx] = 1

        # reshape to coarse grid: (B, rand_size, rand_size)
        mask = mask.reshape(batch_size, self.rand_size, self.rand_size)

        # upscale to model patch resolution
        mask = np.repeat(mask, self.scale, axis=1)
        mask = np.repeat(mask, self.scale, axis=2)

        # return tensor (B, H, W)
        return torch.from_numpy(mask)