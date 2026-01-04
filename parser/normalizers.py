"""
Normalizers Module

This module handles normalization of extracted values to consistent formats.
Normalization is the bridge between messy extracted data and clean CSV output.

What normalization does:
- Dates → ISO format (YYYY-MM-DD)
- Currency → Decimal numbers without symbols
- Names → Proper case, trimmed
- Addresses → Multi-line cleaned up
- Numbers → Consistent decimal format
- Phone numbers → Standard format
- Empty values → NULL representation

Why this matters:
Data extracted from PDFs comes in hundreds of variations:
- "01/15/24", "January 15, 2024", "15-Jan-2024" → "2024-01-15"
- "$1,234.56", "USD 1234.56", "1.234,56€" → 1234.56
- "JOHN SMITH", "john smith" → "John Smith"

Downstream systems (databases, analytics) need consistent formats.
"""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Union

from loguru import logger


class TextNormalizer:
    """Normalizes text values."""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text.
        
        - Collapse multiple spaces to single space
        - Normalize line endings
        - Strip leading/trailing whitespace
        """
        if not text:
            return ""
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Collapse multiple spaces (but preserve newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        
        # Remove space at start/end of lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Collapse multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_case(text: str, style: str = 'title') -> str:
        """
        Normalize text case.
        
        Args:
            text: Input text
            style: 'title', 'upper', 'lower', 'sentence'
        """
        if not text:
            return ""
        
        if style == 'title':
            # Title case, but preserve acronyms and handle special cases
            words = text.split()
            result = []
            
            for word in words:
                # Keep all-caps short words (likely acronyms)
                if word.isupper() and len(word) <= 4:
                    result.append(word)
                # Keep words with mixed case that aren't all upper (McDonald, etc.)
                elif not word.isupper() and any(c.isupper() for c in word[1:]):
                    result.append(word)
                else:
                    result.append(word.title())
            
            return ' '.join(result)
        
        elif style == 'upper':
            return text.upper()
        
        elif style == 'lower':
            return text.lower()
        
        elif style == 'sentence':
            # Capitalize first letter of each sentence
            sentences = re.split(r'([.!?]+\s*)', text)
            result = []
            for i, part in enumerate(sentences):
                if i % 2 == 0 and part:  # Actual sentence content
                    result.append(part[0].upper() + part[1:].lower() if part else '')
                else:
                    result.append(part)
            return ''.join(result)
        
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep: str = '') -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            keep: String of special chars to keep (e.g., '-_.')
        """
        if not text:
            return ""
        
        # Build pattern of chars to keep
        keep_escaped = re.escape(keep)
        pattern = f'[^a-zA-Z0-9\\s{keep_escaped}]'
        
        return re.sub(pattern, '', text)
    
    @staticmethod
    def truncate(text: str, max_length: int, suffix: str = '...') -> str:
        """Truncate text to max length, adding suffix if truncated."""
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix


class DateNormalizer:
    """Normalizes date values to ISO format."""
    
    def __init__(self, formats: Optional[list[str]] = None):
        """
        Initialize date normalizer.
        
        Args:
            formats: List of input date formats to try (in order of priority)
        """
        self.formats = formats or [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%B %d %Y",
            "%b %d %Y",
            "%m-%d-%Y",
            "%m-%d-%y",
            "%d/%m/%y",
            "%m/%d/%y",
        ]
    
    def normalize(
        self, 
        value: str, 
        output_format: str = "%Y-%m-%d"
    ) -> Optional[str]:
        """
        Normalize a date string to standard format.
        
        Args:
            value: Input date string
            output_format: Desired output format (default ISO)
            
        Returns:
            Normalized date string or None if parsing fails
        """
        if not value:
            return None
        
        value = str(value).strip()
        
        # Try each format
        for fmt in self.formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime(output_format)
            except ValueError:
                continue
        
        # Try dateutil as fallback
        try:
            from dateutil import parser as date_parser
            parsed = date_parser.parse(value, fuzzy=True)
            return parsed.strftime(output_format)
        except Exception:
            pass
        
        logger.warning(f"Could not parse date: {value}")
        return None
    
    def extract_and_normalize(self, text: str) -> Optional[str]:
        """
        Extract a date from text and normalize it.
        
        Useful when date is embedded in other text.
        """
        # Common date patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                normalized = self.normalize(match.group(0))
                if normalized:
                    return normalized
        
        return None


class CurrencyNormalizer:
    """Normalizes currency/money values."""
    
    # Currency symbol to code mapping
    SYMBOL_TO_CODE = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '₹': 'INR',
        'Fr': 'CHF',
        'kr': 'SEK',
        'R$': 'BRL',
    }
    
    def normalize(
        self, 
        value: str,
        decimal_places: int = 2
    ) -> Optional[float]:
        """
        Normalize a currency value to a number.
        
        Handles:
        - Currency symbols ($, €, etc.)
        - Thousands separators (1,234 or 1.234)
        - Different decimal separators (1.50 or 1,50)
        - Accounting format ((100) for negative)
        
        Args:
            value: Input currency string
            decimal_places: Number of decimal places in output
            
        Returns:
            Normalized float value or None
        """
        if not value:
            return None
        
        value = str(value).strip()
        
        # Check for negative (accounting format)
        is_negative = '(' in value and ')' in value
        
        # Remove currency symbols and codes
        cleaned = re.sub(r'[^\d.,\-\s]', '', value)
        cleaned = cleaned.strip()
        
        if not cleaned:
            return None
        
        # Handle different number formats
        cleaned = self._normalize_number_format(cleaned)
        
        try:
            result = float(cleaned)
            if is_negative:
                result = -abs(result)
            return round(result, decimal_places)
        except ValueError:
            logger.warning(f"Could not parse currency: {value}")
            return None
    
    def _normalize_number_format(self, value: str) -> str:
        """
        Handle different thousand/decimal separator conventions.
        
        - US/UK: 1,234.56 (comma=thousands, dot=decimal)
        - Europe: 1.234,56 (dot=thousands, comma=decimal)
        - Some: 1 234,56 (space=thousands, comma=decimal)
        """
        # Remove spaces (space as thousands separator)
        value = value.replace(' ', '')
        
        # Handle parentheses (accounting negative)
        value = value.replace('(', '-').replace(')', '')
        
        # Count separators
        dots = value.count('.')
        commas = value.count(',')
        
        if dots == 0 and commas == 0:
            # No separators, return as-is
            return value
        
        if dots == 1 and commas == 0:
            # Single dot - likely decimal separator (1234.56)
            return value
        
        if commas == 1 and dots == 0:
            # Single comma - could be decimal (European) or thousands
            # Check position to decide
            comma_pos = value.index(',')
            after_comma = len(value) - comma_pos - 1
            
            if after_comma <= 2:
                # Likely decimal separator (123,45)
                return value.replace(',', '.')
            else:
                # Likely thousands separator (1,234)
                return value.replace(',', '')
        
        if dots > 0 and commas > 0:
            # Both present - determine which is decimal
            last_dot = value.rfind('.')
            last_comma = value.rfind(',')
            
            if last_comma > last_dot:
                # European format (1.234,56)
                return value.replace('.', '').replace(',', '.')
            else:
                # US format (1,234.56)
                return value.replace(',', '')
        
        if dots > 1:
            # Multiple dots - must be thousands separator (1.234.567)
            return value.replace('.', '')
        
        if commas > 1:
            # Multiple commas - must be thousands separator (1,234,567)
            return value.replace(',', '')
        
        return value
    
    def extract_currency_code(self, value: str) -> Optional[str]:
        """
        Extract currency code from a value string.
        
        Returns 3-letter ISO currency code if found.
        """
        value = str(value).upper()
        
        # Check for ISO codes
        codes = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR']
        for code in codes:
            if code in value:
                return code
        
        # Check for symbols
        for symbol, code in self.SYMBOL_TO_CODE.items():
            if symbol in value:
                return code
        
        return None


class NumberNormalizer:
    """Normalizes numeric values."""
    
    def normalize(
        self, 
        value: str,
        decimal_places: Optional[int] = None,
        as_integer: bool = False
    ) -> Optional[Union[int, float]]:
        """
        Normalize a number value.
        
        Args:
            value: Input number string
            decimal_places: Round to this many decimals (None = keep original)
            as_integer: Force conversion to integer
            
        Returns:
            Normalized number or None
        """
        if not value:
            return None
        
        value = str(value).strip()
        
        # Remove common formatting
        cleaned = re.sub(r'[^\d.,\-]', '', value)
        
        # Handle different separators
        currency_norm = CurrencyNormalizer()
        cleaned = currency_norm._normalize_number_format(cleaned)
        
        try:
            if as_integer:
                return int(float(cleaned))
            
            result = float(cleaned)
            
            if decimal_places is not None:
                result = round(result, decimal_places)
            
            return result
            
        except ValueError:
            logger.warning(f"Could not parse number: {value}")
            return None
    
    def normalize_percentage(self, value: str) -> Optional[float]:
        """
        Normalize a percentage value.
        
        Returns decimal form (e.g., "25%" → 0.25)
        """
        if not value:
            return None
        
        value = str(value).strip()
        
        # Remove % sign
        cleaned = value.replace('%', '').strip()
        
        number = self.normalize(cleaned)
        
        if number is not None:
            return number / 100
        
        return None


class AddressNormalizer:
    """Normalizes address fields."""
    
    def normalize(self, value: str) -> str:
        """
        Normalize an address string.
        
        - Clean up whitespace
        - Standardize line breaks
        - Remove duplicate lines
        - Proper capitalization
        """
        if not value:
            return ""
        
        # Normalize whitespace
        value = TextNormalizer.normalize_whitespace(value)
        
        # Split into lines
        lines = [line.strip() for line in value.split('\n') if line.strip()]
        
        # Remove duplicate consecutive lines
        deduped = []
        for line in lines:
            if not deduped or line.lower() != deduped[-1].lower():
                deduped.append(line)
        
        # Apply title case to each line (but preserve postal codes, etc.)
        normalized_lines = []
        for line in deduped:
            # Check if line looks like a postal code / zip
            if re.match(r'^[\dA-Z]{3,10}[-\s]?[\dA-Z]*$', line.upper()):
                normalized_lines.append(line.upper())
            else:
                normalized_lines.append(TextNormalizer.normalize_case(line, 'title'))
        
        return '\n'.join(normalized_lines)


class InvoiceFieldNormalizer:
    """
    Complete normalizer for invoice fields.
    Combines all specialized normalizers.
    """
    
    def __init__(self, date_formats: Optional[list[str]] = None):
        """Initialize with optional custom date formats."""
        self.text_norm = TextNormalizer()
        self.date_norm = DateNormalizer(formats=date_formats)
        self.currency_norm = CurrencyNormalizer()
        self.number_norm = NumberNormalizer()
        self.address_norm = AddressNormalizer()
    
    def normalize_field(
        self, 
        value: Any, 
        field_type: str,
        field_name: Optional[str] = None
    ) -> Any:
        """
        Normalize a field value based on its type.
        
        Args:
            value: Raw field value
            field_type: Type string (string, date, currency, number, text_block)
            field_name: Optional field name for type-specific handling
            
        Returns:
            Normalized value (or None/empty for invalid)
        """
        if value is None:
            return None
        
        if field_type == 'date':
            return self.date_norm.normalize(str(value))
        
        elif field_type == 'currency':
            return self.currency_norm.normalize(str(value))
        
        elif field_type == 'number':
            return self.number_norm.normalize(str(value))
        
        elif field_type == 'text_block':
            # For addresses or multi-line text
            if field_name and 'address' in field_name.lower():
                return self.address_norm.normalize(str(value))
            return self.text_norm.normalize_whitespace(str(value))
        
        else:  # 'string' or default
            normalized = self.text_norm.normalize_whitespace(str(value))
            
            # Apply title case to name fields
            if field_name and 'name' in field_name.lower():
                normalized = self.text_norm.normalize_case(normalized, 'title')
            
            return normalized
    
    def normalize_all(
        self, 
        data: dict[str, Any], 
        field_types: dict[str, str]
    ) -> dict[str, Any]:
        """
        Normalize all fields in a data dictionary.
        
        Args:
            data: Dict of field_name → raw_value
            field_types: Dict of field_name → field_type
            
        Returns:
            Dict of field_name → normalized_value
        """
        normalized = {}
        
        for field_name, value in data.items():
            field_type = field_types.get(field_name, 'string')
            normalized[field_name] = self.normalize_field(
                value, field_type, field_name
            )
        
        return normalized


# Convenience functions

def normalize_date(value: str, formats: Optional[list[str]] = None) -> Optional[str]:
    """Normalize a date value to ISO format."""
    return DateNormalizer(formats=formats).normalize(value)


def normalize_currency(value: str) -> Optional[float]:
    """Normalize a currency value to a float."""
    return CurrencyNormalizer().normalize(value)


def normalize_number(value: str) -> Optional[float]:
    """Normalize a number value to a float."""
    return NumberNormalizer().normalize(value)


def normalize_text(value: str) -> str:
    """Normalize text whitespace and encoding."""
    return TextNormalizer.normalize_whitespace(value)
