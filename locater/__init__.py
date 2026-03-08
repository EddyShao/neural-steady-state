"""Locater algorithms.

This package groups the previous top-level modules:
- locater_flexible.py -> locater.flexible
- locater_strict.py   -> locater.strict

The old top-level modules remain as import/CLI shims for backward compatibility.
"""

from .flexible import adaptive_peak_detection  # noqa: F401
from .strict import adaptive_peak_detection_amr  # noqa: F401
