from src.constants import *
from src.utils.commons import read_yaml, create_directories
from src.base.config_entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
        self,
        config_path = CONFIG_FILE_PATH):

        self.config = read_yaml(config_path)
        

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.cache_dir])

        data_ingestion_config = DataIngestionConfig(
            cache_dir = Path(PROJECT_BASE_DIR / config.cache_dir),
            data_source = Path(PROJECT_BASE_DIR / config.data_source),
            batch_size= config.batch_size,
            image_size= config.image_size
        )
        return data_ingestion_config