"""
Pluggable Document Types

This package provides configurable document type definitions and extractors.
Supports invoices, bank statements, receipts, resumes, and custom document types.
"""

from .document_type import (
    DocumentType,
    FieldDefinition,
    FieldType,
    DocumentConfig,
)
from .registry import (
    DocumentTypeRegistry,
    register_document_type,
    get_document_type,
)
from .extractors import (
    DocumentExtractor,
    ExtractionResult,
)
from .builtin_types import (
    INVOICE_TYPE,
    BANK_STATEMENT_TYPE,
    RECEIPT_TYPE,
    RESUME_TYPE,
)

__all__ = [
    'DocumentType',
    'FieldDefinition',
    'FieldType',
    'DocumentConfig',
    'DocumentTypeRegistry',
    'register_document_type',
    'get_document_type',
    'DocumentExtractor',
    'ExtractionResult',
    'INVOICE_TYPE',
    'BANK_STATEMENT_TYPE',
    'RECEIPT_TYPE',
    'RESUME_TYPE',
]
