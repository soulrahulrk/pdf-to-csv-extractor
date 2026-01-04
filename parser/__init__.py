"""
Parser Package

This package handles parsing of extracted text into structured data fields.
It includes:
- Field mapping from text to structured fields
- Validation of extracted values
- Normalization to standard formats

Usage:
    from parser import FieldMapper, FieldValidator, InvoiceFieldNormalizer
    
    # Extract fields from text
    mapper = FieldMapper(config_path=Path("config/fields.yaml"))
    fields = mapper.extract_fields(text)
    
    # Validate extracted values
    validator = FieldValidator()
    result = validator.validate_field('invoice_date', '2024-01-15', 'date')
    
    # Normalize values
    normalizer = InvoiceFieldNormalizer()
    normalized = normalizer.normalize_field('$1,234.56', 'currency')
"""

from .field_mapper import (
    FieldMapper,
    ConfigLoader,
    FieldDefinition,
    ExtractedField,
    extract_fields_from_text,
)

from .validators import (
    FieldValidator,
    ValidationResult,
    InvoiceFieldValidator,
    validate_extracted_data,
)

from .normalizers import (
    TextNormalizer,
    DateNormalizer,
    CurrencyNormalizer,
    NumberNormalizer,
    AddressNormalizer,
    InvoiceFieldNormalizer,
    normalize_date,
    normalize_currency,
    normalize_number,
    normalize_text,
)

__all__ = [
    # Field Mapper
    'FieldMapper',
    'ConfigLoader',
    'FieldDefinition',
    'ExtractedField',
    'extract_fields_from_text',
    
    # Validators
    'FieldValidator',
    'ValidationResult',
    'InvoiceFieldValidator',
    'validate_extracted_data',
    
    # Normalizers
    'TextNormalizer',
    'DateNormalizer',
    'CurrencyNormalizer',
    'NumberNormalizer',
    'AddressNormalizer',
    'InvoiceFieldNormalizer',
    'normalize_date',
    'normalize_currency',
    'normalize_number',
    'normalize_text',
]
