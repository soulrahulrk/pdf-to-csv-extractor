"""
PDF to CSV Extractor

A production-grade tool for extracting structured data from PDF documents.

Features:
- Text extraction from digital PDFs (pdfplumber)
- OCR for scanned documents (Tesseract)
- Table extraction (Camelot)
- Field parsing with regex patterns
- Data validation and normalization
- CSV export

Quick Start:
    from pdf_to_csv import extract_from_pdf
    
    # Simple usage
    result = extract_from_pdf('invoice.pdf')
    print(result.fields)
    
    # Or use the Streamlit UI
    # streamlit run app.py

CLI Usage:
    python main.py invoice.pdf --output result.csv
    python main.py ./documents/ --output results.csv --batch
"""

__version__ = '1.0.0'
__author__ = 'PDF to CSV Team'

# Main extractors
from extractor import (
    PDFTextExtractor,
    OCRExtractor,
    TableExtractor,
    ExtractionResult,
    ExtractionMethod,
    remove_noise,
    merge_extraction_results,
    get_pdf_info,
    check_tesseract_installed,
)

# Parsing
from parser import (
    FieldMapper,
    ConfigLoader,
    FieldValidator,
    InvoiceFieldNormalizer,
)

# Output
from output import (
    CSVWriter,
    CSVConfig,
    write_to_csv,
)

__all__ = [
    # Version
    '__version__',
    
    # Extractors
    'PDFTextExtractor',
    'OCRExtractor',
    'TableExtractor',
    'ExtractionResult',
    'ExtractionMethod',
    'remove_noise',
    'merge_extraction_results',
    'get_pdf_info',
    'check_tesseract_installed',
    
    # Parsing
    'FieldMapper',
    'ConfigLoader',
    'FieldValidator',
    'InvoiceFieldNormalizer',
    
    # Output
    'CSVWriter',
    'CSVConfig',
    'write_to_csv',
]
