import torch
from typing import Dict
from omegaconf import OmegaConf


def build_memomae_from_config(config, ModelClass):
    model = ModelClass(config)
    return model
    

def load_backbone_from_ckpt(
    config,
    ModelClass,
    model_key_prefix: str = "model.",
    map_location: str = "cpu",
):
    """
    Load memoMAE weights from a Lightning checkpoint that contained a LightningModule
    with an attribute `self.model` holding memoMAE(config).
    """
    ckpt = torch.load(config.ckpt_path, map_location=map_location, weights_only=False)
    state_dict: Dict[str, torch.Tensor] = ckpt["state_dict"]

    # Strip 'model.' prefix (or whatever you used) so it matches memoMAE's state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(model_key_prefix):
            new_k = k[len(model_key_prefix):]  # remove 'model.'
        else:
            new_k = k
        new_state_dict[new_k] = v

    model = build_memomae_from_config(config, ModelClass)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Loaded memoMAE. Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model
