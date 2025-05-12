import tensorflow as tf
from typing import Tuple
from zenml.steps import step
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer
from src.base.config_entity import DataIngestionConfig
from src.log import logger

@step(
    enable_cache=False,
    output_materializers={
        "output_0": TensorflowDatasetMaterializer,
        "output_1": TensorflowDatasetMaterializer,
        "output_2": TensorflowDatasetMaterializer,
    }
)
def ingestion(data_ingestion_config: DataIngestionConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load images from directories and return train, validation, and test tf.data.Dataset objects."""

    def load_dataset(path: str) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='int',
            image_size=data_ingestion_config.image_size,
            batch_size=data_ingestion_config.batch_size,
            shuffle=True
        )

    try:
        logger.info("Loading datasets from directories...")
        train_ds = load_dataset(str(data_ingestion_config.data_source / "train"))
        val_ds = load_dataset(str(data_ingestion_config.data_source / "val"))
        test_ds = load_dataset(str(data_ingestion_config.data_source / "test"))

        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        logger.success("Datasets loaded and optimized successfully.")

        return train_ds, val_ds, test_ds

    except Exception as e:
        logger.error(f"Failed to load datasets from directories: {e}")
        raise e
