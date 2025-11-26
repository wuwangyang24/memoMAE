from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class TrainPipeline(Pipeline):
    def __init__(self, file_list, batch_size, num_threads,
                 device_id, shard_id, num_shards):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.reader = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            pad_last_batch=True,
            stick_to_shard=True,
        )

    def define_graph(self):
        images, labels = self.reader
        # Mixed = CPU stage + GPU decode
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        # GPU augmentations
        images = fn.random_resized_crop(images, size=(224, 224))
        images = fn.flip(images, horizontal=fn.random.coin_flip())
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        return images, labels
    

class ValPipeline(Pipeline):
    def __init__(self, file_list, batch_size, num_threads,
                 device_id, shard_id, num_shards):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.reader = fn.readers.file(
            file_list=file_list,
            random_shuffle=False,
            shuffle_after_epoch=False,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            stick_to_shard=True
        )

    def define_graph(self):
        images, labels = self.reader
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_x=256, resize_y=256)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        return images, labels
    
class TestPipeline(Pipeline):
    pass
    