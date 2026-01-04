"""
Tests for PDF to CSV Extractor

This module contains unit tests for the extraction pipeline components.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.normalizers import (
    DateNormalizer,
    CurrencyNormalizer,
    NumberNormalizer,
    TextNormalizer,
    AddressNormalizer,
)
from parser.validators import FieldValidator, ValidationResult
from parser.field_mapper import FieldMapper, FieldDefinition, ExtractedField
from output.csv_writer import CSVWriter, CSVConfig, ColumnNamer, ValueFormatter
from extractor.utils import (
    normalize_text,
    remove_noise,
    calculate_text_confidence,
    ExtractionMethod,
)


class TestDateNormalizer:
    """Tests for date normalization."""
    
    def setup_method(self):
        self.normalizer = DateNormalizer()
    
    def test_iso_format(self):
        assert self.normalizer.normalize("2024-01-15") == "2024-01-15"
    
    def test_us_format(self):
        assert self.normalizer.normalize("01/15/2024") == "2024-01-15"
    
    def test_european_format(self):
        # Note: This test assumes day/month/year format
        result = self.normalizer.normalize("15/01/2024")
        assert result == "2024-01-15"
    
    def test_long_format(self):
        assert self.normalizer.normalize("January 15, 2024") == "2024-01-15"
    
    def test_short_month(self):
        assert self.normalizer.normalize("Jan 15, 2024") == "2024-01-15"
    
    def test_invalid_date(self):
        assert self.normalizer.normalize("not a date") is None
    
    def test_empty_input(self):
        assert self.normalizer.normalize("") is None
        assert self.normalizer.normalize(None) is None


class TestCurrencyNormalizer:
    """Tests for currency normalization."""
    
    def setup_method(self):
        self.normalizer = CurrencyNormalizer()
    
    def test_simple_number(self):
        assert self.normalizer.normalize("1234.56") == 1234.56
    
    def test_with_dollar_sign(self):
        assert self.normalizer.normalize("$1,234.56") == 1234.56
    
    def test_with_euro_sign(self):
        assert self.normalizer.normalize("€1.234,56") == 1234.56
    
    def test_negative_accounting(self):
        assert self.normalizer.normalize("($100.00)") == -100.00
    
    def test_thousands_separator(self):
        assert self.normalizer.normalize("1,234,567.89") == 1234567.89
    
    def test_european_decimal(self):
        # European format: comma as decimal
        assert self.normalizer.normalize("1234,56") == 1234.56
    
    def test_currency_code_extraction(self):
        assert self.normalizer.extract_currency_code("$100 USD") == "USD"
        assert self.normalizer.extract_currency_code("€500") == "EUR"
        assert self.normalizer.extract_currency_code("£250") == "GBP"


class TestNumberNormalizer:
    """Tests for number normalization."""
    
    def setup_method(self):
        self.normalizer = NumberNormalizer()
    
    def test_integer(self):
        assert self.normalizer.normalize("123") == 123.0
    
    def test_decimal(self):
        assert self.normalizer.normalize("123.45") == 123.45
    
    def test_with_thousands(self):
        assert self.normalizer.normalize("1,234") == 1234.0
    
    def test_percentage(self):
        assert self.normalizer.normalize_percentage("25%") == 0.25
        assert self.normalizer.normalize_percentage("8.5%") == 0.085
    
    def test_as_integer(self):
        assert self.normalizer.normalize("123.0", as_integer=True) == 123


class TestTextNormalizer:
    """Tests for text normalization."""
    
    def test_whitespace_normalization(self):
        text = "  Hello   World  \n\n\n  Test  "
        result = TextNormalizer.normalize_whitespace(text)
        assert "   " not in result
        assert result.startswith("Hello")
    
    def test_case_title(self):
        assert TextNormalizer.normalize_case("JOHN SMITH", "title") == "John Smith"
        assert TextNormalizer.normalize_case("john smith", "title") == "John Smith"
    
    def test_case_upper(self):
        assert TextNormalizer.normalize_case("hello", "upper") == "HELLO"
    
    def test_case_lower(self):
        assert TextNormalizer.normalize_case("HELLO", "lower") == "hello"
    
    def test_preserve_acronyms(self):
        result = TextNormalizer.normalize_case("IBM CORPORATION", "title")
        assert "IBM" in result


class TestFieldValidator:
    """Tests for field validation."""
    
    def setup_method(self):
        self.validator = FieldValidator()
    
    def test_validate_string(self):
        result = self.validator.validate_field("name", "John Doe", "string")
        assert result.is_valid
        assert result.validated_value == "John Doe"
    
    def test_validate_date(self):
        result = self.validator.validate_field("date", "2024-01-15", "date")
        assert result.is_valid
        assert result.validated_value == "2024-01-15"
    
    def test_validate_currency(self):
        result = self.validator.validate_field("total", "$1,234.56", "currency")
        assert result.is_valid
        assert result.validated_value == 1234.56
    
    def test_validate_number(self):
        result = self.validator.validate_field("qty", "100", "number")
        assert result.is_valid
        assert result.validated_value == 100.0
    
    def test_invalid_date(self):
        result = self.validator.validate_field("date", "not a date", "date")
        assert not result.is_valid
    
    def test_empty_value(self):
        result = self.validator.validate_field("name", "", "string")
        assert not result.is_valid
    
    def test_cross_validate_total(self):
        # Total should match subtotal + tax
        fields = {
            'subtotal': 100.0,
            'tax_amount': 10.0,
            'grand_total': 110.0
        }
        warnings = self.validator.cross_validate(fields)
        assert len(warnings) == 0
    
    def test_cross_validate_mismatch(self):
        # Mismatched total should generate warning
        fields = {
            'subtotal': 100.0,
            'tax_amount': 10.0,
            'grand_total': 200.0  # Wrong!
        }
        warnings = self.validator.cross_validate(fields)
        assert any("mismatch" in w.lower() for w in warnings)


class TestColumnNamer:
    """Tests for column name processing."""
    
    def test_snake_case(self):
        assert ColumnNamer.to_snake_case("Invoice Number") == "invoice_number"
        assert ColumnNamer.to_snake_case("grandTotal") == "grand_total"
        assert ColumnNamer.to_snake_case("TAX_AMOUNT") == "tax_amount"
    
    def test_clean_column_name(self):
        assert ColumnNamer.clean_column_name("Invoice #") == "Invoice"
        assert ColumnNamer.clean_column_name("123column") == "column"
    
    def test_make_unique(self):
        columns = ["name", "name", "name", "date"]
        result = ColumnNamer.make_unique(columns)
        assert result == ["name", "name_1", "name_2", "date"]


class TestValueFormatter:
    """Tests for value formatting."""
    
    def setup_method(self):
        self.formatter = ValueFormatter(CSVConfig())
    
    def test_format_none(self):
        assert self.formatter.format_value(None) == ""
    
    def test_format_number(self):
        assert self.formatter.format_value(1234.5678, "number") == "1234.57"
    
    def test_format_integer(self):
        assert self.formatter.format_value(1234.0, "number") == "1234"
    
    def test_format_date(self):
        dt = datetime(2024, 1, 15)
        assert self.formatter.format_value(dt, "date") == "2024-01-15"
    
    def test_format_list(self):
        result = self.formatter.format_value(["a", "b", "c"])
        assert result == "a | b | c"


class TestExtractorUtils:
    """Tests for extractor utility functions."""
    
    def test_normalize_text(self):
        text = "Hello\u00a0World"  # Non-breaking space
        result = normalize_text(text)
        assert "\u00a0" not in result
    
    def test_normalize_ligatures(self):
        text = "ﬁnd ﬂow"  # Contains ligatures
        result = normalize_text(text)
        assert "fi" in result
        assert "fl" in result
    
    def test_remove_noise(self):
        text = "Invoice Data\nPage 1 of 10\nMore content"
        patterns = [r'Page \d+ of \d+']
        result = remove_noise(text, patterns)
        assert "Page 1 of 10" not in result
        assert "Invoice Data" in result
    
    def test_calculate_confidence_text(self):
        good_text = "This is a normal sentence with words."
        confidence = calculate_text_confidence(good_text, ExtractionMethod.TEXT_LAYER)
        assert confidence > 0.8
    
    def test_calculate_confidence_garbage(self):
        bad_text = "@#$%^&*(){}[]"
        confidence = calculate_text_confidence(bad_text, ExtractionMethod.TEXT_LAYER)
        assert confidence < 0.7
    
    def test_calculate_confidence_ocr(self):
        text = "Normal text from OCR"
        confidence = calculate_text_confidence(text, ExtractionMethod.OCR)
        # OCR should have lower base confidence
        assert confidence < calculate_text_confidence(text, ExtractionMethod.TEXT_LAYER)


class TestFieldMapper:
    """Tests for field mapping."""
    
    def setup_method(self):
        # Create a simple config for testing
        self.config = {
            'fields': [
                {
                    'name': 'invoice_number',
                    'display_name': 'Invoice Number',
                    'type': 'string',
                    'required': True,
                    'patterns': [r'(?i)invoice\s*#?[:\s]*([A-Z0-9-]+)'],
                    'keywords': ['invoice'],
                },
                {
                    'name': 'total',
                    'display_name': 'Total',
                    'type': 'currency',
                    'required': True,
                    'patterns': [r'(?i)total[:\s]*\$?([\d,]+\.?\d*)'],
                    'keywords': ['total'],
                }
            ],
            'settings': {},
            'noise_patterns': []
        }
        self.mapper = FieldMapper(config=self.config)
    
    def test_extract_invoice_number(self):
        text = "Invoice # INV-2024-001\nSome other content"
        results = self.mapper.extract_fields(text)
        
        inv_result = next((r for r in results if r.name == 'invoice_number'), None)
        assert inv_result is not None
        assert inv_result.is_valid
        assert "INV-2024-001" in inv_result.value
    
    def test_extract_total(self):
        text = "Subtotal: $100.00\nTotal: $110.00"
        results = self.mapper.extract_fields(text)
        
        total_result = next((r for r in results if r.name == 'total'), None)
        assert total_result is not None
        assert total_result.is_valid
    
    def test_missing_field(self):
        text = "No invoice data here"
        results = self.mapper.extract_fields(text)
        
        inv_result = next((r for r in results if r.name == 'invoice_number'), None)
        assert inv_result is not None
        assert not inv_result.is_valid


class TestCSVWriter:
    """Tests for CSV writing."""
    
    def test_to_string(self):
        records = [
            {'name': 'John', 'amount': 100.0},
            {'name': 'Jane', 'amount': 200.0},
        ]
        
        writer = CSVWriter()
        csv_string = writer.to_string(records)
        
        assert 'name' in csv_string
        assert 'amount' in csv_string
        assert 'John' in csv_string
        assert '100' in csv_string
    
    def test_snake_case_columns(self):
        records = [{'Invoice Number': 'INV-001'}]
        
        config = CSVConfig(use_snake_case=True)
        writer = CSVWriter(config=config)
        csv_string = writer.to_string(records)
        
        assert 'invoice_number' in csv_string
    
    def test_null_handling(self):
        records = [{'name': 'John', 'email': None}]
        
        writer = CSVWriter()
        csv_string = writer.to_string(records)
        
        # None should become empty string
        assert 'None' not in csv_string


# Integration test
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_extraction_pipeline(self):
        """Test the full extraction pipeline with sample text."""
        from parser.field_mapper import FieldMapper
        from parser.validators import FieldValidator
        from parser.normalizers import InvoiceFieldNormalizer
        
        # Sample invoice text
        sample_text = """
        INVOICE
        
        Invoice Number: INV-2024-001234
        Date: January 15, 2024
        
        Bill To:
        Acme Corporation
        123 Business St
        New York, NY 10001
        
        Description          Qty    Price    Amount
        Widget A             10     $25.00   $250.00
        Widget B             5      $50.00   $250.00
        
        Subtotal:                           $500.00
        Tax (8%):                           $40.00
        Total Due:                          $540.00
        """
        
        # Configure mapper
        config = {
            'fields': [
                {
                    'name': 'invoice_number',
                    'display_name': 'Invoice Number',
                    'type': 'string',
                    'required': True,
                    'patterns': [r'(?i)invoice\s*(?:number|#)?[:\s]*([A-Z0-9-]+)'],
                    'keywords': ['invoice'],
                },
                {
                    'name': 'invoice_date',
                    'display_name': 'Invoice Date',
                    'type': 'date',
                    'required': True,
                    'patterns': [r'(?i)date[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})'],
                    'keywords': ['date'],
                },
                {
                    'name': 'grand_total',
                    'display_name': 'Total',
                    'type': 'currency',
                    'required': True,
                    'patterns': [r'(?i)total\s*(?:due)?[:\s]*\$?([\d,]+\.?\d*)'],
                    'keywords': ['total'],
                }
            ],
            'settings': {
                'date_formats': ['%B %d, %Y', '%Y-%m-%d']
            },
            'noise_patterns': []
        }
        
        # Extract fields
        mapper = FieldMapper(config=config)
        results = mapper.extract_fields(sample_text)
        
        # Validate
        validator = FieldValidator(date_formats=['%B %d, %Y', '%Y-%m-%d'])
        normalizer = InvoiceFieldNormalizer(date_formats=['%B %d, %Y', '%Y-%m-%d'])
        
        extracted = {}
        for result in results:
            if result.is_valid:
                field_type = next(
                    (f['type'] for f in config['fields'] if f['name'] == result.name),
                    'string'
                )
                validation = validator.validate_field(
                    result.name, result.value, field_type
                )
                if validation.is_valid:
                    normalized = normalizer.normalize_field(
                        validation.validated_value, field_type, result.name
                    )
                    extracted[result.name] = normalized
        
        # Assertions
        assert 'invoice_number' in extracted
        assert 'INV-2024-001234' in str(extracted['invoice_number'])
        
        assert 'invoice_date' in extracted
        assert extracted['invoice_date'] == '2024-01-15'
        
        assert 'grand_total' in extracted
        assert extracted['grand_total'] == 540.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
