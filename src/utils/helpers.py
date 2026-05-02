import time
import functools
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} completed in {end - start:.2f}s")
        return result
    return wrapper

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent