import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

def encode_latents(mae_baseline, 
                   train_loader, 
                   val_loader, 
                   num_neighbors_train: int=5,
                   num_neighbors_val: int=5,
                   # memory for training
                   fill_memory_train: bool=False,
                   rounds_train: int=10,
                   # memory for validation
                   reset_memory: bool=False, #reset memory bank
                   fill_memory_val: bool=False,
                   rounds_val: int=10,
                   device: str='cpu'):
    
    mae_baseline = mae_baseline.to(device)
    mae_baseline.eval()
    
    # --------- Memory bank before training ---------
    if fill_memory_train:
        for _ in tqdm(range(rounds_train), desc="Filling memory bank for training"):
            with torch.no_grad():
                for images, labels in tqdm(train_loader):
                    mae_baseline.forward_encoder_memo(images.to(device), 
                                                      mask_ratio=0., 
                                                      memorize=True, 
                                                      fill_memory=True)
                    
    # --------- Encode latents: train set ---------
    LATENTS_TRAIN = []
    LABELS_TRAIN = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Encoding train set"):
            latents = mae_baseline.forward_encoder_memo(images.to(device), 
                                                        mask_ratio=0., 
                                                        k_sim_patches=num_neighbors_train, 
                                                        memorize=False,
                                                        fill_memory=False
                                                       )[0]
            LATENTS_TRAIN.append(latents.mean(1).cpu())
            LABELS_TRAIN.append(labels.to(torch.long).cpu())

    # --------- Memory bank before validation ---------
    if reset_memory:
        mae_baseline.memory_bank.reset()
        print('memory bank is reset')
    if fill_memory_val:
        for _ in tqdm(range(rounds_val), desc="Filling memory bank for validation"):
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    mae_baseline.forward_encoder_memo(images.to(device), 
                                                      mask_ratio=0., 
                                                      memorize=True, 
                                                      fill_memory=True)
            
    # --------- Encode latents: val set ---------
    LATENTS_VAL = []
    LABELS_VAL = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Encoding val set"):
            latents = mae_baseline.forward_encoder_memo(images.to(device), 
                                                        mask_ratio=0., 
                                                        k_sim_patches=num_neighbors_val, 
                                                        memorize=False,
                                                        fill_memory=False
                                                       )[0]
            LATENTS_VAL.append(latents.mean(1).cpu())
            LABELS_VAL.append(labels.to(torch.long).cpu())
    return LATENTS_TRAIN, LABELS_TRAIN, LATENTS_VAL, LABELS_VAL

    
def linearprobing(LATENTS_TRAIN, LABELS_TRAIN, LATENTS_VAL, LABELS_VAL, device: str='cpu'):
    # ----- 1. Stack features & labels -----
    feats_train = torch.cat(LATENTS_TRAIN, dim=0)   # (N, D)
    labels_train = torch.cat(LABELS_TRAIN, dim=0) # (N,)
    feats_test = torch.cat(LATENTS_VAL, dim=0)   # (N, D)
    labels_test = torch.cat(LABELS_VAL, dim=0) # (N,)
    N, D = feats_train.shape
    num_classes = int(labels_train.max().item()) + 1
    # move to device
    if device != 'cpu':
        feats_train = feats_train.to(device)
        labels_train = labels_train.to(device)
        feats_test = feats_test.to(device)
        labels_test = labels_test.to(device)
    # ----- 3. Linear classifier with BatchNorm -----
    clf = nn.Sequential(
        nn.BatchNorm1d(D, affine=False),
        nn.Linear(D, num_classes)
    ).to(device)
    # MAE LR scaling rule
    batch_size = 2048
    batch_size = min(batch_size, feats_train.size(0))
    base_lr = 0.1
    max_lr = base_lr * batch_size / 256.0
    train_loader = DataLoader(
        TensorDataset(feats_train, labels_train),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.SGD(
        clf.parameters(),
        lr=max_lr,          # final LR, warmup scheduler will scale it
        momentum=0.9,
        weight_decay=0.0,
    )
    # ----- 4. Linear warmup + cosine schedulers -----
    warmup_epochs = 10
    total_epochs = 100
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,       # start at near-zero LR
        end_factor=1.0,          # warm up to max_lr
        total_iters=warmup_epochs,
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
    )
    criterion = nn.CrossEntropyLoss()
    # ----- 5. Training -----
    clf.train()
    with torch.enable_grad():
        for epoch in tqdm(range(total_epochs), desc='Training linear prober'):
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = clf(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
            scheduler.step()
    # ----- 6. Evaluation -----
    clf.eval()
    with torch.no_grad():
        preds = clf(feats_test).argmax(dim=1)
        acc = (preds == labels_test).float().mean().item()
    return acc
    