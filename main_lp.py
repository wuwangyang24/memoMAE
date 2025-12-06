import os
import torch
import argparse
from omegaconf import OmegaConf
from Experiments.utils import load_backbone_from_ckpt
from Experiments.dataloader import ImagenetData
from Model.memoMAE import memoMAE
from Experiments.LinearProbing.linearprobing import encode_latents, linearprobing
torch.set_float32_matmul_precision('medium')

_model_mapping = {'pammae': memoMAE}

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    config = OmegaConf.load(config_path)
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training script for pamvit linear probing')
    parser.add_argument('--config', type=str, default='Configs/config_mae_lp.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # load model ckpt
    print('Loading checkpoint...')
    mae_baseline = load_backbone_from_ckpt(config=config,
                                           ModelClass=_model_mapping[config.model])

    # Initialize DataModule
    print('Creating dataloader...')
    data_module = ImagenetData(train_txt=config['train'],
                               val_txt=config['val'],
                               root_dir=config['root_dir']
                               batch_size=config['batch_size'])

    # Encode latents
    print('Encoding latents...')
    LATENTS = encode_latents(mae_baseline=mae_baseline, 
                             train_loader=data_module.train_dataloader(), 
                             val_loader=data_module.val_dataloader(), 
                             num_neighbors=config.num_neighbors,
                             # memory for training
                             fill_memory_train=config.fill_memory_train,
                             rounds_train=config.rounds_train,
                             # memory for validation
                             reset_memory=config.reset_memory, #reset memory bank
                             fill_memory_val=config.fill_memory_val,
                             rounds_val=config.rounds_val,
                             device=config.encode_device
                            )

    # Linear probing
    print('Linear probing...')
    acc = linearprobing(*LATENTS, device=config.training.clf_device)
    print(acc)

if __name__ == '__main__':
    main()
