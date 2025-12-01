import os
import re
from typing import Optional
import warnings
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
from omegaconf import DictConfig
from Trainer.pl_module import LightningModel
from Model.memoMAE import memoMAE
from Model.memoSimMIM import MemoSimMIM
warnings.filterwarnings("ignore")

_model_map = {'mae': memoMAE, 'simmim': MemoSimMIM}

class Trainer:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.name = self._generate_experiment_name(config)
        self.wandb_logger = self._init_wandb_logger()
        self.model = _model_map[config.model](config)
        self.pl_module = LightningModel(self.model, config)
        self.wandb_logger.watch(self.model, log="gradients", log_freq=1000)
        self.resume_checkpoint = self._find_latest_checkpoint(self._checkpoint_dir)
        self.pl_trainer = self._init_pl_trainer()

    @property
    def _checkpoint_dir(self) -> str:
        """Construct checkpoint directory path."""
        parts = [
            f'{self.config.model}',
            f'ViT-{self.config.size}',
            f'Patch-{self.config.data.patch_size}',
            f'MemoCap{self.config.memory_bank.memory_capacity}', 
            f'NumSim{self.config.hyperparameters.num_neighbors}', 
            f'NosimEpochs{self.config.hyperparameters.nosim_train_epochs}'
        ]
        p_name = "-".join(parts)
        return os.path.join(self.config.checkpoint.save_dir, p_name)

    def _init_wandb_logger(self) -> WandbLogger:
        """Initialize Weights & Biases logger."""
        return WandbLogger(
            log_model=True,
            entity=self.config.logging.entity,
            project=self.config.logging.project,
            name=self.name,
            resume='allow'
        )

    def _generate_experiment_name(self, cfg: DictConfig) -> str:
        """Generate a structured experiment name from configuration."""
        parts = [
            f'{cfg.model}',
            f'ViT-{cfg.size}',
            f'Patch-{cfg.data.patch_size}',
            f'MemoCap{cfg.memory_bank.memory_capacity}', 
            f'NumSim{cfg.hyperparameters.num_neighbors}', 
            f'NosimEpochs{cfg.hyperparameters.nosim_train_epochs}'
        ]
        return "-".join(parts)

    def _find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        '''Find the latest checkpoint file in the specified directory.'''
        def pick_checkpoint(files):
            '''Pick the checkpoint file with the highest epoch number.'''
            numbered = []
            for f in files:
                nums = re.findall(r'\d+', f)
                if nums:  # keep only those with digits
                    numbered.append((int(nums[-1]), f))
            if not numbered:
                return None  # or raise an error
            return max(numbered, key=lambda x: x[0])[1]
        """Return path to latest checkpoint or None if not found."""
        if not os.path.exists(checkpoint_dir):
            print("No checkpoint directory available")
            return None
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if not checkpoints:
                print("No checkpoint files found")
                return None
            latest_ckpt = pick_checkpoint(checkpoints)
            checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
            print(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"Error finding checkpoint: {e}")
            return None

    def _init_pl_trainer(self) -> pl.Trainer:
        """Initialize PyTorch Lightning Trainer with optimized settings."""
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=self._checkpoint_dir,
            filename='checkpoint_{epoch}',
            every_n_epochs=self.config.checkpoint.save_every_epochs,
            save_top_k=1,
            save_last=True,
        )
        profiler = AdvancedProfiler(filename="advanced_profiler")

        return pl.Trainer(
            accelerator="gpu",
            devices=[self.config.device],
            max_epochs=self.config.training.max_epochs,
            gradient_clip_val=1.0,
            accumulate_grad_batches=self.config.training.accumulate_grad_batches,
            check_val_every_n_epoch=self.config.training.val_every_epochs,
            logger=self.wandb_logger,
            log_every_n_steps=1,
            precision="16-mixed",
            num_sanity_val_steps=0,
            callbacks=[
                checkpoint_callback,
                pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
                pl.pytorch.callbacks.ModelSummary(max_depth=4),
                #pl.pytorch.callbacks.DeviceStatsMonitor(),
            ],
            enable_model_summary=True,
            # deterministic=True,
            profiler=profiler,
        )

    def train(self, data_module: pl.LightningDataModule) -> None:
        """Train the model with checkpoint resumption."""
        try:
            print("Starting training...")
            self.pl_trainer.fit(
                model=self.pl_module,
                datamodule=data_module,
                ckpt_path=self.resume_checkpoint
            )
            print("Training completed successfully.")
        except Exception as e:
            print(f"Training failed: {e}")
            # Optionally still dump profiler here:
            if hasattr(self.pl_trainer, "profiler") and hasattr(self.pl_trainer.profiler, "summary"):
                print("Saving profiler after exception...")
                self.pl_trainer.profiler.summary()
            raise
        finally:
            # Always try to dump the profiler at the very end
            if hasattr(self.pl_trainer, "profiler") and hasattr(self.pl_trainer.profiler, "summary"):
                try:
                    print("Final profiler summary...")
                    self.pl_trainer.profiler.summary()
                except Exception as e:
                    print(f"Could not write profiler summary: {e}")
            import wandb
            wandb.finish()
