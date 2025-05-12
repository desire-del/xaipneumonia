from src.constants import *
from src.utils.commons import read_yaml, create_directories
from src.base.config_entity import DataIngestionConfig, DataPreprocessingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_path = CONFIG_FILE_PATH):

        self.config = read_yaml(config_path)
        

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.cache_dir])

        data_ingestion_config = DataIngestionConfig(
            cache_dir = str(PROJECT_BASE_DIR / config.cache_dir),
            data_source = str(PROJECT_BASE_DIR / config.data_source)
        )
        return data_ingestion_config
    
    def get_data_preprocess_config(self):
        config = self.config.data_processing

        data_augmentation_config = DataPreprocessingConfig(
            normalize=config.normalize,
            augment=config.augment,
            batch_size=config.batch_size,
            image_size=config.image_size,
            augmentation_config=config.augmentation_config
        )
        return data_augmentation_config