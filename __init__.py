"""
Document Intelligence Pipeline

A production-grade document processing system for extracting structured data from PDFs.

Features:
- Layout-aware extraction with spatial reasoning
- Region-based OCR with intelligent preprocessing
- Semantic validation and arithmetic checking
- Categorical decision system (VERIFIED/LIKELY/REVIEW_REQUIRED/REJECTED)
- Multi-page table stitching
- Human-in-the-loop review mode
- Performance hardening for scale
- Pluggable document types
- ML-ready dataset export

Quick Start:
    from pdf_to_csv import process_pdf, DocumentIntelligencePipeline
    
    # Simple usage
    result = process_pdf('invoice.pdf')
    print(result.fields)
    print(result.document_decision.decision)  # VERIFIED, LIKELY, etc.
    
    # Advanced usage
    from pdf_to_csv import PipelineConfig, ProcessingMode
    
    config = PipelineConfig(
        mode=ProcessingMode.THOROUGH,
        enable_ocr=True,
        enable_table_detection=True,
    )
    pipeline = DocumentIntelligencePipeline(config)
    result = pipeline.process_document('invoice.pdf', document_type='invoice')

CLI Usage:
    # Process single file
    python -m pdf_to_csv.cli process invoice.pdf -o output.json
    
    # Process directory
    python -m pdf_to_csv.cli process ./documents/ -o results.csv -f csv
    
    # List available document types
    python -m pdf_to_csv.cli list-types
    
    # Detect document type
    python -m pdf_to_csv.cli detect document.pdf
"""

__version__ = '2.0.0'
__author__ = 'Document Intelligence Team'

# Main pipeline
from .pipeline import (
    DocumentIntelligencePipeline,
    PipelineConfig,
    ProcessingMode,
    ProcessingMetrics,
    DocumentResult,
    process_pdf,
    process_batch,
)

# Decision system
from .decision.decision_engine import (
    Decision,
    DecisionReason,
    FieldDecision,
    DocumentDecision,
    DecisionEngine,
)

# Document types
from .doctypes.document_type import (
    DocumentType,
    FieldDefinition,
    FieldType,
    DocumentConfig,
    create_field,
)
from .doctypes.registry import (
    DocumentTypeRegistry,
    register_document_type,
    get_document_type,
    list_document_types,
    detect_document_type,
)
from .doctypes.builtin_types import (
    INVOICE_TYPE,
    BANK_STATEMENT_TYPE,
    RECEIPT_TYPE,
    RESUME_TYPE,
    register_builtin_types,
)
from .doctypes.extractors import (
    DocumentExtractor,
    ExtractionResult,
    ExtractedValue,
    ExtractionMethod,
)

# Layout analysis
from .layout.box import BoundingBox, TextBlock, BlockType
from .layout.spatial_index import SpatialIndex, Direction
from .layout.layout_graph import LayoutGraph, RelationType
from .layout.analyzer import LayoutAnalyzer, PageLayout, DocumentLayout

# Validation
from .validation.semantic_rules import SemanticValidator, ValidationIssue
from .validation.arithmetic_checks import ArithmeticChecker

# Tables
from .tables.table_detector import TableDetector, TableBoundary
from .tables.table_stitcher import TableStitcher, StitchedTable

# Review
from .review.review_data import ReviewSession, ReviewField, ReviewStatus
from .review.html_preview import HTMLPreviewGenerator
from .review.json_export import ReviewExporter

# Dataset export
from .dataset_export import (
    DatasetExporter,
    ExportFormat,
    DocumentAnnotation,
    DatasetSplit,
    create_dataset_from_results,
)

# Initialize built-in types on import
register_builtin_types()

__all__ = [
    # Version
    '__version__',
    
    # Main pipeline
    'DocumentIntelligencePipeline',
    'PipelineConfig',
    'ProcessingMode',
    'ProcessingMetrics',
    'DocumentResult',
    'process_pdf',
    'process_batch',
    
    # Decision system
    'Decision',
    'DecisionReason',
    'FieldDecision',
    'DocumentDecision',
    'DecisionEngine',
    
    # Document types
    'DocumentType',
    'FieldDefinition',
    'FieldType',
    'DocumentConfig',
    'create_field',
    'DocumentTypeRegistry',
    'register_document_type',
    'get_document_type',
    'list_document_types',
    'detect_document_type',
    'INVOICE_TYPE',
    'BANK_STATEMENT_TYPE',
    'RECEIPT_TYPE',
    'RESUME_TYPE',
    'register_builtin_types',
    'DocumentExtractor',
    'ExtractionResult',
    'ExtractedValue',
    'ExtractionMethod',
    
    # Layout
    'BoundingBox',
    'TextBlock',
    'BlockType',
    'SpatialIndex',
    'Direction',
    'LayoutGraph',
    'RelationType',
    'LayoutAnalyzer',
    'PageLayout',
    'DocumentLayout',
    
    # Validation
    'SemanticValidator',
    'ValidationIssue',
    'ArithmeticChecker',
    
    # Tables
    'TableDetector',
    'TableBoundary',
    'TableStitcher',
    'StitchedTable',
    
    # Review
    'ReviewSession',
    'ReviewField',
    'ReviewStatus',
    'HTMLPreviewGenerator',
    'ReviewExporter',
    
    # Dataset
    'DatasetExporter',
    'ExportFormat',
    'DocumentAnnotation',
    'DatasetSplit',
    'create_dataset_from_results',
]
