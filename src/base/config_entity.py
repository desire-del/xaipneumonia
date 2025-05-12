from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from src.log import logger

class DataIngestionConfig(BaseModel):
    data_source: str
    cache_dir: str
    
    

class DataAugmentationConfig(BaseModel):
    featurewise_center: bool
    samplewise_center: bool
    featurewise_std_normalization:bool
    samplewise_std_normalization: bool
    zca_whitening: bool
    rotation_range: int = Field(..., ge=0, le=180)
    zoom_range : float = Field(..., ge=0.0, le=1.0)
    width_shift_range :float = Field(..., ge=0.0, le=1.0)
    height_shift_range :float = Field(..., ge=0.0, le=1.0)
    horizontal_flip : bool
    vertical_flip : bool

class DataPreprocessingConfig(BaseModel):
    normalize: bool
    augment: bool
    batch_size: int
    image_size: list
    augmentation_config: Optional[DataAugmentationConfig] = None

    @model_validator(mode='after')
    def check_augmentation_config_required(cls, values):
        if values.augment and values.augmentation_config is None:
            logger.error("augmentation_config must be provided when augment is True")
            raise ValueError("augmentation_config must be provided when augment is True")
        return values