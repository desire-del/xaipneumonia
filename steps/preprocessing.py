from typing import Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from zenml.steps import step
from src.base.config_entity import DataPreprocessingConfig
from pathlib import Path
from typing import Annotated
from src.log import logger
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer

def convert_to_tf_dataset(directory_iterator, image_size):
    """Convert a directory iterator to a tf.data.Dataset."""
    def generator():
        for batch, labels in directory_iterator:
            yield batch, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32)
        )
    )
    return dataset


def preprocessing_step(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    config: DataPreprocessingConfig,
):
    """Apply preprocessing to datasets and return train, val, test generators."""

    # Data Augmentation
    if config.augment:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255 if config.normalize else None,
            **config.augmentation_config.model_dump(exclude_unset=True)
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255 if config.normalize else None,
        )

    # Creating the generators
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        class_mode='categorical'
    )
    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical'
    )
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Converting to TensorFlow Dataset and applying optimizations
    def optimize_dataset(dataset, buffer_size=1000):
        # Cache data in memory if possible (especially useful for small datasets)
        dataset = dataset.cache()

        # Shuffle before batching
        dataset = dataset.shuffle(buffer_size)

        # Prefetch data for later processing while training
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Batch the data
        dataset = dataset.batch(config.batch_size)
        return dataset

    # Convert generators to TensorFlow datasets
    train_dataset = convert_to_tf_dataset(train_gen, config.image_size)
    val_dataset = convert_to_tf_dataset(val_gen, config.image_size)
    test_dataset = convert_to_tf_dataset(test_gen, config.image_size)

    # Apply optimizations
    train_dataset = optimize_dataset(train_dataset)
    val_dataset = optimize_dataset(val_dataset)
    test_dataset = optimize_dataset(test_dataset)

    logger.info("Preprocessing completed successfully")

    return train_dataset, val_dataset, test_dataset
