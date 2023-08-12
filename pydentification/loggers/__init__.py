import logging

from .extended_logger import FAIL, NOTIFY, STOP, SUCCESS, ExtendedLogger


def get_logger(name: str) -> logging.Logger:
    """
    Creates instance of logger with custom levels and formatting
    Sets logger level to DEBUG without propagation to child loggers
    """
    logging.setLoggerClass(ExtendedLogger)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()

        # blue formatting for logger name
        formatter = logging.Formatter(f"\033[94m{name}:\033[0m %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger


# allow importing const values for logging levels and get_logger handle
__all__ = ["FAIL", "NOTIFY", "STOP", "SUCCESS", "get_logger"]
