import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging() -> None:
    """
    Configure basic application logging.

    Logging helps us see important events:
    - application startup
    - dataset loading
    - model training
    - predictions
    - errors
    """

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for a specific module.

    Example:
        logger = get_logger(__name__)
    """

    return logging.getLogger(name)