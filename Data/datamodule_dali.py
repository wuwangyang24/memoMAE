from nvidia.dali.plugin.pytorch import DALIGenericIterator
from .pipeline_dali import TrainPipeline, ValPipeline, TestPipeline
import lightning as pl


class DALIDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_list, val_list, test_list=None,
                 batch_size=256, num_threads=12):
        super().__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list  # can be None
        self.batch_size = batch_size
        self.num_threads = num_threads

    def setup(self, stage=None):
        # Lightning requires this function, but pipelines
        # must be constructed inside dataloader methods.
        pass

    def _create_dali_loader(self, pipeline_cls, file_list):
        device_id = self.trainer.local_rank
        world_size = self.trainer.world_size
        pipe = pipeline_cls(
            file_list=file_list,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=device_id,
            shard_id=device_id,
            num_shards=world_size,
        )
        pipe.build()
        return DALIGenericIterator(
            pipelines=pipe,
            output_map=["images", "labels"],  # Returned keys
            auto_reset=True,
            reader_name="Reader"
        )

    def train_dataloader(self):
        return self._create_dali_loader(TrainPipeline, self.train_list)

    def val_dataloader(self):
        return self._create_dali_loader(ValPipeline, self.val_list)

    def test_dataloader(self):
        if self.test_list is None:
            return None
        return self._create_dali_loader(TestPipeline, self.test_list)