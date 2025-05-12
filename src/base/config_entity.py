from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    data_source: Path
    cache_dir: Path
    batch_size: int
    image_size: list