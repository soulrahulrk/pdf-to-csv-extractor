"""
Multi-Page Table Stitching

This package provides intelligent table extraction across page boundaries.
"""

from .table_detector import (
    TableDetector,
    TableBoundary,
    TableType,
)
from .table_stitcher import (
    TableStitcher,
    StitchingConfig,
    StitchedTable,
    TableRow,
    TableCell,
)
from .continuation_detector import (
    ContinuationDetector,
    ContinuationSignal,
    ContinuationConfidence,
)

__all__ = [
    'TableDetector',
    'TableBoundary',
    'TableType',
    'TableStitcher',
    'StitchingConfig',
    'StitchedTable',
    'TableRow',
    'TableCell',
    'ContinuationDetector',
    'ContinuationSignal',
    'ContinuationConfidence',
]
