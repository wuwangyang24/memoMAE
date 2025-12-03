import os
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

def simple_collate(batch: List[Tuple[torch.Tensor, int]]):
    """Batch is a list of (image, label)."""
    imgs, labels = zip(*batch)                 # tuples of length B
    imgs = torch.stack(imgs, dim=0)           # (B, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

class TxtImageDataset(Dataset):
    """
    Dataset for a txt file with lines: 'path class_id'.
    Example line:
        /data/imagenet/train/n02119789/n02119789_0001.JPEG  0
    or:
        n02119789/n02119789_0001.JPEG  0    (plus a root_dir)
    """

    def __init__(
        self,
        txt_file: str,
        root_dir: str = "",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by whitespace: last token = class, rest = path
                parts = line.split()
                path = " ".join(parts[:-1])     # supports spaces in paths
                label = int(parts[-1])
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        img_path = rel_path
        if self.root_dir and not os.path.isabs(rel_path):
            img_path = os.path.join(self.root_dir, rel_path)

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImagenetData:
    """
    Simple helper to build train/val datasets and dataloaders
    from ImageNet-style txt files, without Lightning.
    """

    def __init__(
        self,
        train_txt: str,
        val_txt: str,
        root_dir: str = "",
        batch_size: int = 256,
        num_workers: int = 8,
        img_size: int = 224,
    ):
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = 0
        self.img_size = img_size

        # Define transforms (adjust as needed)
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None

    def setup(self):
        """Construct datasets. Call once before creating dataloaders."""
        if self.train_txt is not None:
            self._train_dataset = TxtImageDataset(
                txt_file=self.train_txt,
                root_dir=self.root_dir,
                transform=self.train_transform,
            )
            
        if self.val_txt is not None:
            self._val_dataset = TxtImageDataset(
                txt_file=self.val_txt,
                root_dir=self.root_dir,
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            collate_fn=simple_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup()
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
            collate_fn=simple_collate,
        )
