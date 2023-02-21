"""Top-level package for Conditional KDE."""

__author__ = """David Prelogović"""
__email__ = "david.prelogovic@gmail.com"
__version__ = "0.1.0"

from .gaussian import ConditionalGaussian, ConditionalGaussianKernelDensity
from .interpolated import InterpolatedConditionalGaussian, InterpolatedConditionalKernelDensity

__all__ = [
    "ConditionalGaussian",
    "ConditionalGaussianKernelDensity",
    "InterpolatedConditionalGaussian",
    "InterpolatedConditionalKernelDensity",
]
