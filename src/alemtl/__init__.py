"""ALEMTL package."""

from importlib.metadata import PackageNotFoundError, version

from .models.multitask_model import MultiTaskModel

try:
    __version__ = version("alemtl")
except PackageNotFoundError:
    # Allows imports directly from an unpackaged source checkout.
    __version__ = "0+unknown"

__all__ = ["MultiTaskModel", "__version__"]