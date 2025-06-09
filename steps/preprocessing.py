from typing import Tuple, Dict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from pathlib import Path
from src.base.config_entity import DataPreprocessingConfig
from src.log import logger

def convert_to_tf_dataset(directory_iterator, image_size: Tuple[int, int]) -> tf.data.Dataset:
    """Convert a directory iterator to a tf.data.Dataset."""
    def generator():
        for batch, labels in directory_iterator:
            yield batch, labels

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)  # <-- Correction ici
        )
    )
    return dataset

def optimize_dataset(dataset: tf.data.Dataset, shuffle: bool = True) -> tf.data.Dataset:
    """Optimize a dataset with optional shuffling, caching and prefetching."""
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    return dataset.cache().prefetch(tf.data.AUTOTUNE)

def preprocessing_step(
    train_dir: Path,
    val_dir: Path,
    config: DataPreprocessingConfig
):
    """Apply preprocessing using ImageDataGenerator with or without augmentation."""

    # Préparer la configuration d'augmentation
    datagen_config = config.augmentation_config.model_dump(exclude_unset=True) if config.augment else {}

    datagen = ImageDataGenerator(
        rescale=1.0 / 255 if config.normalize else None,
        **datagen_config
    )

    # Création des générateurs
    train_gen = datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=True,
        color_mode='rgb',
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        shuffle=False,
        color_mode='rgb',
        class_mode='binary'
    )

    # Récupération des classes
    class_indices: Dict[str, int] = train_gen.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    class_names = [index_to_class[i] for i in sorted(index_to_class)]

    logger.info(f"Classes détectées: {class_names}")
    logger.info(f"Mapping classe (0/1): {index_to_class}")

    # Conversion en tf.data.Dataset
    #train_dataset = convert_to_tf_dataset(train_gen, config.image_size)
    #val_dataset = convert_to_tf_dataset(val_gen, config.image_size)

    # Optimisation
    #train_dataset = optimize_dataset(train_dataset, shuffle=True)
    #val_dataset = optimize_dataset(val_dataset, shuffle=False)

    logger.info("Prétraitement terminé avec succès.")

    return train_gen, val_gen, index_to_class
