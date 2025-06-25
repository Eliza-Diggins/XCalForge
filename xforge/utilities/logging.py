""" Logging utilities for the XCalForge ecosystem.
"""
import logging

from .config import xcfconfig

# ======================== #
# Configure the global log #
# ======================== #
spec_logger = logging.getLogger("SpecForge")
spec_logger.setLevel(
    getattr(logging, xcfconfig["logging.main.level"])
)  # Allow DEBUG, handlers filter final output
spec_logger.propagate = False  # Avoid duplicate logs to root logger

# Don't permit double handler adding.
if not spec_logger.hasHandlers():
    # Console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_fmt = xcfconfig["logging.main.format"]
    console_handler.setFormatter(logging.Formatter(console_fmt))
    spec_logger.addHandler(console_handler)


# ================================ #
# Logger Management.               #
# ================================ #
class RankFormatter(logging.Formatter):
    """
    Custom formatter to inject MPI rank into all log messages.
    """

    def __init__(self, fmt=None, datefmt=None, rank=0):
        super().__init__(fmt, datefmt)
        self.rank = rank

    def format(self, record):
        record.rank = self.rank
        return super().format(record)
