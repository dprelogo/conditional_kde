"""Top-level package for Conditional KDE."""

__author__ = """David Prelogović"""
__email__ = "david.prelogovic@gmail.com"
__version__ = "0.1.0"

from .gaussian import ConditionalGaussianKernelDensity
from .interpolated import InterpolatedConditionalKernelDensity

__all__ = [
    "ConditionalGaussianKernelDensity",
    "InterpolatedConditionalKernelDensity",
]
