from loguru import logger
import sys
from pathlib import Path
from src.constants import PROJECT_BASE_DIR

# Define log directory and ensure it exists
LOG_DIR = PROJECT_BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define log file path with timestamp
log_file = LOG_DIR / "app_{time:YYYY-MM-DD}.log"

# Configure logger
logger.remove()  # Remove default handler

# Console output (stdout)
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    colorize=True
)

# File output
logger.add(
    str(log_file),
    level="INFO",  # INFO and above (INFO, WARNING, ERROR, etc.)
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="00:00",         # Rotate daily
    retention="7 days",       # Keep logs for 7 days
    compression="zip"         # Compress old logs
)
