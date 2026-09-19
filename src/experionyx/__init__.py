"""EXPERIONYX: AI Experimental Forensics & Reliability Laboratory."""

import logging
from importlib.metadata import version

__version__ = version("experionyx")

# Library convention: emit nothing unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
