from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


class TrainPipeline(Pipeline):
    def __init__(
        self,
        file_list,
        batch_size,
        num_threads,
        device_id,
        shard_id,
        num_shards,
        base_seed=42,
        prefetch_queue_depth=4,
    ):
        seed = int(base_seed) + int(shard_id)
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
            exec_async=True,
            exec_pipelined=True,
        )

        self.input = fn.readers.file(
            file_list=file_list,
            random_shuffle=True,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            pad_last_batch=True,      # keep batches full; DALI is happier
            stick_to_shard=True,
            prefetch_queue_depth=prefetch_queue_depth,
        )

    def define_graph(self):
        images, labels = self.input

        # Decode on GPU (mixed) + standard ImageNet-style aug
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.random_resized_crop(
            images,
            size=(224, 224),
        )
        mirror = fn.random.coin_flip(probability=0.5)

        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror,
        )
        return images, labels


class ValPipeline(Pipeline):
    def __init__(
        self,
        file_list,
        batch_size,
        num_threads,
        device_id,
        shard_id,
        num_shards,
        base_seed=42,
        prefetch_queue_depth=4,
    ):
        seed = int(base_seed) + int(shard_id)
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth,
            exec_async=True,
            exec_pipelined=True,
        )

        self.input = fn.readers.file(
            file_list=file_list,
            random_shuffle=False,
            shuffle_after_epoch=False,
            shard_id=shard_id,
            num_shards=num_shards,
            name="Reader",
            pad_last_batch=True,
            stick_to_shard=True,
            prefetch_queue_depth=prefetch_queue_depth,
        )

    def define_graph(self):
        images, labels = self.input

        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=256)
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=0,
        )
        return images, labels


class TestPipeline(ValPipeline):
    # For ImageNet-style eval, test == val
    pass
