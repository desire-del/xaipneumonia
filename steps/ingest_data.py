import tensorflow as tf
from typing import Tuple
from zenml.steps import step
from src.base.config_entity import DataIngestionConfig
from src.log import logger
from typing import Annotated
from pathlib import Path



@step
def ingestion(data_ingestion_config: DataIngestionConfig) -> Tuple[Annotated[Path, "train_dir"], Annotated[Path, "val_dir"], Annotated[Path, "test_dir"]]:
    """Load images from directories and return train, validation, and test tf.data.Dataset objects."""
    logger.info("Data ingested successfully")
    try:
        logger.info("Data ingested successfully")
        train_dir = Path(data_ingestion_config.data_source) / "train"
        val_dir = Path(data_ingestion_config.data_source) / "val"
        test_dir = Path(data_ingestion_config.data_source) / "test"
        logger.info(f"Train directory: {train_dir}")
        logger.info(f"Validation directory: {val_dir}")
        logger.info(f"Test directory: {test_dir}")
        return train_dir, val_dir, test_dir
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise e
