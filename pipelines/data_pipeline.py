from zenml import pipeline
from steps.ingest_data import ingestion
from steps.preprocessing import preprocessing_step
from src.utils.configuration import ConfigurationManager
from typing import Tuple
import tensorflow as tf
from src.log import logger


#@pipeline
def data_pipeline(ingestion_config, data_preprocess_config):
    """Pipeline to ingest data and return train, test, validation datasets."""
    try:
        train_dir, val_dir, test_dir = ingestion(ingestion_config)
        train_ds, val_ds, index_to_class = preprocessing_step(train_dir, val_dir, data_preprocess_config)
        logger.info("Data pipeline completed successfully.")
        return train_ds, val_ds, index_to_class
    except Exception as e:
        print(f"Error in data pipeline: {e}")


    