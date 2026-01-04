"""
Human-in-the-Loop Review System

This package provides tools for human review of extraction results.
Supports JSON export with bounding boxes and HTML preview generation.
"""

from .review_data import (
    ReviewItem,
    ReviewField,
    ReviewStatus,
    ReviewSession,
    ReviewDecision,
)
from .html_preview import (
    HTMLPreviewGenerator,
    PreviewConfig,
)
from .json_export import (
    ReviewExporter,
    ExportFormat,
)

__all__ = [
    'ReviewItem',
    'ReviewField',
    'ReviewStatus',
    'ReviewSession',
    'ReviewDecision',
    'HTMLPreviewGenerator',
    'PreviewConfig',
    'ReviewExporter',
    'ExportFormat',
]
