import logging
import os
import resource
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


def setup_logging(level: int = logging.INFO, quiet_dependencies=True) -> None:
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
        for name in (
            "datasets",
            "PIL",
            "torch",
            "fsspec",
            "urllib3",
            "httpx",
            "httpcore",
            "filelock",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


def add_file_handler(path: str | Path) -> logging.Handler:
    """Add a DEBUG-level, line-buffered file handler to the root logger.

    Uses ``buffering=1`` (line-buffered) so every log record is flushed to
    disk immediately.  This ensures that a SIGKILL (e.g. OOM killer) cannot
    discard buffered output that was already emitted by the application.

    Returns the handler so the caller can remove it later if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Bare open() is intentional: the stream's lifetime is tied to the handler,
    # not to a local scope, so a context manager would close it prematurely.
    stream = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return handler


def log_memory_usage(logger: logging.Logger, label: str) -> None:
    """Log current and peak RSS plus system-available memory at DEBUG level.

    Current RSS is read from ``/proc/self/status`` (Linux only).
    Peak RSS comes from :func:`resource.getrusage` (POSIX, reported by the
    kernel in kilobytes on Linux).  System-available memory is read from
    ``/proc/meminfo`` (``MemAvailable``).

    Silently does nothing on platforms where that information is not available.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    current_kb = int(line.split()[1])
                    break
            else:
                return
    except (FileNotFoundError, OSError):
        return

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux

    available_gb = ""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    available_gb = f", system available: {avail_kb / 1_048_576:.1f} GB"
                    break
    except (FileNotFoundError, OSError):
        pass

    logger.debug(
        "[memory] %s — RSS: %.1f GB, peak RSS: %.1f GB%s",
        label,
        current_kb / 1_048_576,
        peak_kb / 1_048_576,
        available_gb,
    )
