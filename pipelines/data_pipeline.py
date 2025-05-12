from zenml import pipeline
from steps.ingest_data import ingestion
from src.base.config_entity import DataIngestionConfig
from typing import Tuple
import tensorflow as tf


@pipeline
def data_pipeline(config: DataIngestionConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Pipeline to ingest data and return train, test, validation datasets."""
    data = ingestion(config)
    

    