import os
from typing import Optional, Callable, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import lightning as pl


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

        # Read the list once; this is fine even for very large datasets
        # (millions of lines) because it's only storing strings + ints.
        self.samples: List[Tuple[str, int]] = []
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by whitespace: last token = class, rest = path
                parts = line.split()
                # In case paths contain spaces, join everything except last
                path = " ".join(parts[:-1])
                label = int(parts[-1])
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        img_path = rel_path
        if self.root_dir and not os.path.isabs(rel_path):
            img_path = os.path.join(self.root_dir, rel_path)
        # Lazy load image (important for huge datasets)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImagenetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_txt: str,
        val_txt: str,
        root_dir: str = "",
        batch_size: int = 256,
        num_workers: int = 8,
        img_size: int = 224,
    ):
        super().__init__()
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        # Define transforms (adjust as you like)
        self.train_transform = transforms.Compose([
            transforms.Resize(int(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(int(img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Called by Lightning on each process; cheap to construct datasets here
        if stage in (None, "fit"):
            self._train_dataset = TxtImageDataset(
                txt_file=self.train_txt,
                root_dir=self.root_dir,
                transform=self.train_transform,
            )
            self._val_dataset = TxtImageDataset(
                txt_file=self.val_txt,
                root_dir=self.root_dir,
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
