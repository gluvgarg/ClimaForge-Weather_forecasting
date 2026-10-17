import logging

from src.utils.constants import LOG_DIR


def get_logger(name):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "weather_system.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(name)
