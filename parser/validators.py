"""
Validators Module

This module provides validation for extracted field values using Pydantic
for type coercion and custom validation rules.

Why validation matters:
- Extracted data can be garbage (OCR errors, pattern mismatches)
- Downstream systems expect clean, properly typed data
- Catching errors early prevents corrupted databases
- Validation confidence helps prioritize manual review

Validation levels:
1. Type validation (is it a valid date? number? etc.)
2. Format validation (does the date use expected format?)
3. Business rule validation (is the amount reasonable?)
4. Cross-field validation (does total = subtotal + tax?)
"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator, ValidationError
from loguru import logger


@dataclass
class ValidationResult:
    """Result of validating a single field."""
    field_name: str
    original_value: Any
    validated_value: Any
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # Positive or negative adjustment


class InvoiceFieldValidator(BaseModel):
    """
    Pydantic model for invoice field validation.
    
    This model defines expected types and validation rules for
    common invoice fields. Extend this class for different document types.
    """
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    vendor_name: Optional[str] = None
    subtotal: Optional[str] = None
    tax_amount: Optional[str] = None
    tax_rate: Optional[str] = None
    discount_amount: Optional[str] = None
    grand_total: Optional[str] = None
    payment_terms: Optional[str] = None
    purchase_order: Optional[str] = None
    currency: Optional[str] = None
    
    @field_validator('invoice_number')
    @classmethod
    def validate_invoice_number(cls, v):
        """Invoice numbers should be non-empty alphanumeric strings."""
        if v is None:
            return None
        v = str(v).strip()
        if len(v) < 2:
            raise ValueError('Invoice number too short')
        if len(v) > 50:
            raise ValueError('Invoice number too long')
        return v
    
    @field_validator('customer_name', 'vendor_name')
    @classmethod
    def validate_name(cls, v):
        """Names should contain at least some letters."""
        if v is None:
            return None
        v = str(v).strip()
        if not any(c.isalpha() for c in v):
            raise ValueError('Name must contain letters')
        return v
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v):
        """Currency should be 3-letter code or symbol."""
        if v is None:
            return None
        v = str(v).strip().upper()
        valid_currencies = [
            'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 
            'INR', 'MXN', 'BRL', 'KRW', 'SGD', 'HKD', 'NOK', 'SEK'
        ]
        if len(v) == 3 and v.isalpha():
            return v
        if v in ['$', '€', '£', '¥', '₹']:
            # Map symbols to codes
            symbol_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR'}
            return symbol_map.get(v, v)
        raise ValueError(f'Invalid currency: {v}')


class FieldValidator:
    """
    Validates extracted field values against expected types and rules.
    
    Usage:
        validator = FieldValidator()
        result = validator.validate_field('invoice_date', '2024-01-15', 'date')
        if result.is_valid:
            print(result.validated_value)  # datetime object
        else:
            print(result.errors)
    """
    
    def __init__(self, date_formats: Optional[list[str]] = None):
        """
        Initialize validator.
        
        Args:
            date_formats: List of date formats to try when parsing dates
        """
        self.date_formats = date_formats or [
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
        ]
    
    def validate_field(
        self, 
        field_name: str, 
        value: Any, 
        field_type: str,
        validation_rules: Optional[dict] = None
    ) -> ValidationResult:
        """
        Validate a single field value.
        
        Args:
            field_name: Name of the field
            value: Extracted value to validate
            field_type: Expected type (string, number, currency, date, text_block)
            validation_rules: Optional additional validation rules
            
        Returns:
            ValidationResult with validation outcome
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=None,
                is_valid=False,
                errors=["Value is empty"]
            )
        
        # Type-specific validation
        validators = {
            'string': self._validate_string,
            'number': self._validate_number,
            'currency': self._validate_currency,
            'date': self._validate_date,
            'text_block': self._validate_text_block,
        }
        
        validator_func = validators.get(field_type, self._validate_string)
        result = validator_func(field_name, value)
        
        # Apply additional validation rules
        if validation_rules and result.is_valid:
            result = self._apply_rules(result, validation_rules)
        
        return result
    
    def _validate_string(self, field_name: str, value: Any) -> ValidationResult:
        """Validate string fields."""
        str_value = str(value).strip()
        
        # Check for garbage content
        garbage_chars = sum(1 for c in str_value if ord(c) < 32 and c not in '\n\t')
        
        if garbage_chars > len(str_value) * 0.1:
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=str_value,
                is_valid=False,
                errors=["Too many garbage characters"],
                confidence_adjustment=-0.3
            )
        
        return ValidationResult(
            field_name=field_name,
            original_value=value,
            validated_value=str_value,
            is_valid=True
        )
    
    def _validate_number(self, field_name: str, value: Any) -> ValidationResult:
        """Validate numeric fields."""
        errors = []
        
        # Clean the value
        str_value = str(value).strip()
        
        # Remove common formatting
        cleaned = str_value.replace(',', '').replace(' ', '')
        cleaned = cleaned.replace('(', '-').replace(')', '')  # Handle accounting format
        
        try:
            # Try to convert to Decimal for precision
            num_value = Decimal(cleaned)
            
            # Check for reasonable range
            if abs(num_value) > Decimal('999999999999'):
                errors.append("Number seems unreasonably large")
            
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=float(num_value),
                is_valid=True,
                warnings=errors
            )
            
        except (InvalidOperation, ValueError) as e:
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=None,
                is_valid=False,
                errors=[f"Cannot parse as number: {str_value}"]
            )
    
    def _validate_currency(self, field_name: str, value: Any) -> ValidationResult:
        """Validate currency/money fields."""
        import re
        
        str_value = str(value).strip()
        
        # Extract numeric part, removing currency symbols
        cleaned = re.sub(r'[^\d.,\-]', '', str_value)
        
        # Handle different decimal separators
        # European format: 1.234,56 -> 1234.56
        if ',' in cleaned and '.' in cleaned:
            if cleaned.rfind(',') > cleaned.rfind('.'):
                # European format
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Could be European decimal or thousands separator
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                # Likely decimal (e.g., "100,50")
                cleaned = cleaned.replace(',', '.')
            else:
                # Likely thousands separator
                cleaned = cleaned.replace(',', '')
        
        try:
            amount = Decimal(cleaned)
            
            # Basic sanity checks
            warnings = []
            if amount < 0:
                warnings.append("Negative amount")
            if amount > Decimal('9999999999'):
                warnings.append("Amount seems very large")
            
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=float(amount),
                is_valid=True,
                warnings=warnings
            )
            
        except (InvalidOperation, ValueError):
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=None,
                is_valid=False,
                errors=[f"Cannot parse as currency: {str_value}"]
            )
    
    def _validate_date(self, field_name: str, value: Any) -> ValidationResult:
        """Validate date fields."""
        from dateutil import parser as date_parser
        
        str_value = str(value).strip()
        
        # Try configured formats first
        for fmt in self.date_formats:
            try:
                parsed = datetime.strptime(str_value, fmt)
                
                # Sanity check: date should be reasonable
                warnings = []
                if parsed.year < 1990:
                    warnings.append("Date seems very old")
                if parsed.year > 2100:
                    warnings.append("Date seems far in the future")
                
                return ValidationResult(
                    field_name=field_name,
                    original_value=value,
                    validated_value=parsed.strftime("%Y-%m-%d"),
                    is_valid=True,
                    warnings=warnings
                )
            except ValueError:
                continue
        
        # Fall back to dateutil parser (more flexible but less predictable)
        try:
            parsed = date_parser.parse(str_value, fuzzy=True)
            
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=parsed.strftime("%Y-%m-%d"),
                is_valid=True,
                warnings=["Date parsed with fuzzy matching"],
                confidence_adjustment=-0.1
            )
            
        except (ValueError, OverflowError):
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=None,
                is_valid=False,
                errors=[f"Cannot parse as date: {str_value}"]
            )
    
    def _validate_text_block(self, field_name: str, value: Any) -> ValidationResult:
        """Validate multi-line text blocks (addresses, descriptions)."""
        str_value = str(value).strip()
        
        # Clean up excessive whitespace
        import re
        str_value = re.sub(r'\n{3,}', '\n\n', str_value)
        str_value = re.sub(r' {3,}', ' ', str_value)
        
        # Check for reasonable content
        if len(str_value) < 5:
            return ValidationResult(
                field_name=field_name,
                original_value=value,
                validated_value=str_value,
                is_valid=False,
                errors=["Text block too short"]
            )
        
        return ValidationResult(
            field_name=field_name,
            original_value=value,
            validated_value=str_value,
            is_valid=True
        )
    
    def _apply_rules(
        self, 
        result: ValidationResult, 
        rules: dict
    ) -> ValidationResult:
        """Apply additional validation rules."""
        value = result.validated_value
        errors = list(result.errors)
        warnings = list(result.warnings)
        
        # Min/max length for strings
        if isinstance(value, str):
            if 'min_length' in rules and len(value) < rules['min_length']:
                errors.append(f"Value too short (min {rules['min_length']})")
            if 'max_length' in rules and len(value) > rules['max_length']:
                errors.append(f"Value too long (max {rules['max_length']})")
        
        # Min/max for numbers
        if isinstance(value, (int, float)):
            if 'min_value' in rules and value < rules['min_value']:
                errors.append(f"Value below minimum ({rules['min_value']})")
            if 'max_value' in rules and value > rules['max_value']:
                errors.append(f"Value above maximum ({rules['max_value']})")
        
        # Pattern matching
        if 'pattern' in rules and isinstance(value, str):
            import re
            if not re.match(rules['pattern'], value):
                errors.append(f"Value doesn't match expected pattern")
        
        return ValidationResult(
            field_name=result.field_name,
            original_value=result.original_value,
            validated_value=value,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_adjustment=result.confidence_adjustment
        )
    
    def validate_all(
        self, 
        fields: dict[str, tuple[Any, str]],
        validation_rules: Optional[dict[str, dict]] = None
    ) -> dict[str, ValidationResult]:
        """
        Validate multiple fields at once.
        
        Args:
            fields: Dict of {field_name: (value, field_type)}
            validation_rules: Optional dict of {field_name: rules_dict}
            
        Returns:
            Dict of {field_name: ValidationResult}
        """
        results = {}
        
        for field_name, (value, field_type) in fields.items():
            rules = (validation_rules or {}).get(field_name, {})
            results[field_name] = self.validate_field(
                field_name, value, field_type, rules
            )
        
        return results
    
    def cross_validate(
        self, 
        fields: dict[str, Any]
    ) -> list[str]:
        """
        Perform cross-field validation (business logic checks).
        
        Examples:
        - Total should equal subtotal + tax - discount
        - Due date should be after invoice date
        - Tax amount should be reasonable % of subtotal
        
        Args:
            fields: Dict of validated field values
            
        Returns:
            List of validation warning messages
        """
        warnings = []
        
        # Check total calculation
        subtotal = fields.get('subtotal')
        tax = fields.get('tax_amount', 0)
        discount = fields.get('discount_amount', 0)
        total = fields.get('grand_total')
        
        if all(v is not None for v in [subtotal, total]):
            try:
                subtotal = float(subtotal) if subtotal else 0
                tax = float(tax) if tax else 0
                discount = float(discount) if discount else 0
                total = float(total) if total else 0
                
                calculated_total = subtotal + tax - discount
                
                # Allow 1% tolerance for rounding
                if abs(calculated_total - total) > total * 0.01 + 1:
                    warnings.append(
                        f"Total mismatch: calculated {calculated_total:.2f} "
                        f"vs extracted {total:.2f}"
                    )
            except (ValueError, TypeError):
                pass
        
        # Check date logic
        invoice_date = fields.get('invoice_date')
        due_date = fields.get('due_date')
        
        if invoice_date and due_date:
            try:
                inv_dt = datetime.strptime(invoice_date, "%Y-%m-%d")
                due_dt = datetime.strptime(due_date, "%Y-%m-%d")
                
                if due_dt < inv_dt:
                    warnings.append("Due date is before invoice date")
                    
                if (due_dt - inv_dt).days > 365:
                    warnings.append("Due date is more than a year after invoice date")
            except ValueError:
                pass
        
        # Check tax rate reasonableness
        if subtotal and tax:
            try:
                subtotal_val = float(subtotal)
                tax_val = float(tax)
                
                if subtotal_val > 0:
                    tax_rate = (tax_val / subtotal_val) * 100
                    
                    if tax_rate > 50:
                        warnings.append(f"Tax rate seems high ({tax_rate:.1f}%)")
                    elif tax_rate < 0:
                        warnings.append("Negative tax amount")
            except (ValueError, TypeError, ZeroDivisionError):
                pass
        
        return warnings


def validate_extracted_data(
    data: dict[str, Any], 
    field_types: dict[str, str]
) -> tuple[dict[str, Any], list[str]]:
    """
    Convenience function to validate extracted data.
    
    Args:
        data: Dict of {field_name: value}
        field_types: Dict of {field_name: type_string}
        
    Returns:
        Tuple of (validated_data, error_messages)
    """
    validator = FieldValidator()
    validated = {}
    errors = []
    
    for field_name, value in data.items():
        field_type = field_types.get(field_name, 'string')
        result = validator.validate_field(field_name, value, field_type)
        
        if result.is_valid:
            validated[field_name] = result.validated_value
        else:
            errors.extend([f"{field_name}: {e}" for e in result.errors])
    
    return validated, errors
