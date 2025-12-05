from nvidia.dali.plugin.pytorch import DALIGenericIterator
from .pipeline_dali import TrainPipeline, ValPipeline, TestPipeline
import lightning as pl
import os


def _filelist_len(path):
    # One line = one sample. Assumes standard DALI file_list format.
    with open(path, "r") as f:
        return sum(1 for _ in f)


class DALIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_list,
        val_list,
        test_list=None,
        batch_size=256,
        num_threads=8,
        prefetch_queue_depth=4,
    ):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.prefetch_queue_depth = prefetch_queue_depth

        # Will be created in setup()
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    # ------------------------------------------------------------------
    # Rank / world helpers
    # ------------------------------------------------------------------
    def _get_rank(self):
        # Lightning usually sets LOCAL_RANK per process (DDP)
        return int(os.environ.get("LOCAL_RANK", 0))

    def _get_world(self):
        return int(os.environ.get("WORLD_SIZE", 1))

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage=None):
        # Build once and reuse – building pipelines every call is expensive.
        rank = self._get_rank()
        world_size = self._get_world()

        if stage in (None, "fit"):
            train_size = _filelist_len(self.train_list)
            val_size = _filelist_len(self.val_list)

            train_pipe = TrainPipeline(
                file_list=self.train_list,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=rank,
                shard_id=rank,
                num_shards=world_size,
                prefetch_queue_depth=self.prefetch_queue_depth,
            )
            val_pipe = ValPipeline(
                file_list=self.val_list,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=rank,
                shard_id=rank,
                num_shards=world_size,
                prefetch_queue_depth=self.prefetch_queue_depth,
            )

            train_pipe.build()
            val_pipe.build()

            self._train_loader = DALIGenericIterator(
                pipelines=[train_pipe],
                output_map=["images", "labels"],
                auto_reset=True,
                reader_name="Reader",
            )
            self._val_loader = DALIGenericIterator(
                pipelines=[val_pipe],
                output_map=["images", "labels"],
                auto_reset=True,
                reader_name="Reader",
            )

        if stage in (None, "test") and self.test_list is not None:
            test_size = _filelist_len(self.test_list)
            test_pipe = TestPipeline(
                file_list=self.test_list,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=rank,
                shard_id=rank,
                num_shards=world_size,
                prefetch_queue_depth=self.prefetch_queue_depth,
            )
            test_pipe.build()

            self._test_loader = DALIGenericIterator(
                pipelines=[test_pipe],
                output_map=["images", "labels"],
                size=test_size // world_size,
                auto_reset=True,
                reader_name="Reader",
            )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader
