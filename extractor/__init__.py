"""
Extractor Package

This package provides PDF content extraction capabilities with multiple
strategies for handling different PDF types (digital, scanned, tables).

The extractor package exposes three main classes:
- PDFTextExtractor: For extracting text from PDF text layers
- OCRExtractor: For extracting text from scanned images/PDFs
- TableExtractor: For extracting tabular data from PDFs

Usage:
    from extractor import PDFTextExtractor, OCRExtractor, TableExtractor
    
    # Text extraction
    text_ext = PDFTextExtractor()
    result = text_ext.extract_text(Path("document.pdf"))
    
    # OCR for scanned documents
    ocr_ext = OCRExtractor(language='eng')
    result = ocr_ext.extract_from_pdf(Path("scanned.pdf"))
    
    # Table extraction
    table_ext = TableExtractor()
    tables = table_ext.extract_tables(Path("invoice.pdf"))
"""

from .pdf_text import PDFTextExtractor, extract_text_simple
from .ocr import OCRExtractor, check_tesseract_installed
from .tables import TableExtractor, extract_tables_simple
from .utils import (
    ExtractionResult,
    ExtractionMethod,
    PageAnalysis,
    normalize_text,
    remove_noise,
    calculate_text_confidence,
    merge_extraction_results,
    get_pdf_info,
)

__all__ = [
    # Main extractors
    'PDFTextExtractor',
    'OCRExtractor',
    'TableExtractor',
    
    # Data structures
    'ExtractionResult',
    'ExtractionMethod',
    'PageAnalysis',
    
    # Utility functions
    'normalize_text',
    'remove_noise',
    'calculate_text_confidence',
    'merge_extraction_results',
    'get_pdf_info',
    
    # Convenience functions
    'extract_text_simple',
    'extract_tables_simple',
    'check_tesseract_installed',
]
