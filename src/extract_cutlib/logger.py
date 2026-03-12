from __future__ import annotations

import logging
import os


def get_logger(logger_path: str = "./log/cutLib_extraction.log", name: str = "cutLibLogger") -> logging.Logger:
    log_dir = os.path.dirname(logger_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(logger_path, "w", encoding="utf-8"):
        pass

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(logger_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
