import logging
import os
from pathlib import Path

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """
    Console handler that routes output through ``tqdm.write()``.

    When a tqdm progress bar is active, ``tqdm.write()`` clears the bar line,
    prints the message, then redraws the bar.  When no bar is active it
    degrades to a normal ``sys.stderr.write()``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logging(level: int = logging.INFO, quiet_dependencies=False) -> None:
    """
    Configure the root logger for the whole process.

    Call once at CLI entry.

    Console goes through :class:`TqdmLoggingHandler` at *level*.
    ``LOG_LEVEL`` env-var overrides the console level
      (e.g. ``LOG_LEVEL=DEBUG uv run ...``).

    Noisy third-party loggers can be silenced to WARNING.
    """
    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if env_level:
        level = getattr(logging, env_level, level)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers decide what to show

    console = TqdmLoggingHandler()
    console.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(formatter)
    root.addHandler(console)

    # Quiet down chatty libraries.
    if quiet_dependencies:
        for name in ("datasets", "PIL", "torch", "fsspec", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


def add_file_handler(path: str | Path) -> logging.FileHandler:
    """Add a DEBUG-level file handler to the root logger.

    Returns the handler so the caller can remove it later if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return handler
