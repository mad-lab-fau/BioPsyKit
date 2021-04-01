"""
Top level package for biopsykit.
"""

import biopsykit.signals
import biopsykit.saliva
import biopsykit.sleep
import biopsykit.protocols
import biopsykit.questionnaires
import biopsykit.metadata
import biopsykit.io
import biopsykit.carwatch_logs
import biopsykit.classification
import biopsykit.plotting
import biopsykit.stats

import biopsykit.colors
import biopsykit.example_data
import biopsykit.utils

__all__ = [
    "signals",
    "saliva",
    "sleep",
    "stats",
    "protocols",
    "questionnaires",
    "metadata",
    "classification",
    "io",
    "carwatch_logs",
    "colors",
    "example_data",
    "utils",
]

# Info
__version__ = "0.0.1"

# Maintainer info
__author__ = "Robert Richer"
__email__ = "robert.richer@fau.de"
