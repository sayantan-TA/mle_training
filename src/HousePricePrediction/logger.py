import logging
import sys


def setup_logger(
    log_level: int = logging.DEBUG,
    log_to_file: bool = False,
    log_file_path: str = None,
    console_log: bool = True,
) -> logging.Logger:
    """
    Set up a logger with specified configurations.

    Parameters:
    - log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    - log_to_file (bool): If True, log messages will be written to a file.
    - log_file_path (Optional[str]): Path to the log file. Required if log_to_file is True.
    - console_log (bool): If True, log messages will be printed to the console.

    Returns:
    - logging.Logger: Configured logger instance.

    Example:
    >>> logger = setup_logger(log_level=logging.INFO, log_to_file=True,\
        log_file_path="/path/to/log.log", console_log=False)
    >>> logger.info("This is an informational message.")
    This is an informational message.
    """
    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create a console handler and set the level to DEBUG if console_log is True
    if console_log:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # Create a file handler and set the level to DEBUG if log_to_file is True
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # Create a formatter and attach it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if console_log:
        console_handler.setFormatter(formatter)
    if log_to_file:
        file_handler.setFormatter(formatter)

    return logger
