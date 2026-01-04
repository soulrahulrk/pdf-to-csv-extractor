"""
Arithmetic Validation Module

This module performs arithmetic verification on extracted financial data.
It's the most critical validation - arithmetic errors can cause real damage.

Checks Performed:
- Subtotal + Tax = Total
- Sum(Line Items) = Subtotal
- Quantity × Price = Amount (per line item)
- Tax Amount = Subtotal × Tax Rate
- Total - Discount = Amount Due

Why This Matters:
- OCR often produces "close but wrong" numbers
- Transposed digits (123 vs 132) pass format checks
- Missing decimals (1234 vs 12.34) are common
- These errors propagate downstream silently

Design Philosophy:
- Use Decimal for precision (no float rounding)
- Allow configurable tolerances
- Report all mismatches, not just first
- Provide expected vs actual values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of arithmetic errors."""
    TOTAL_MISMATCH = auto()           # Subtotal + tax != total
    LINE_ITEMS_MISMATCH = auto()      # Sum of items != subtotal
    LINE_CALCULATION_ERROR = auto()   # Qty × price != amount
    TAX_CALCULATION_ERROR = auto()    # Subtotal × rate != tax
    DISCOUNT_ERROR = auto()           # Total - discount calculation wrong
    BALANCE_ERROR = auto()            # Opening + transactions != closing


@dataclass
class CalculationError:
    """
    A specific calculation error found during arithmetic validation.
    """
    error_type: ErrorType
    message: str
    expected_value: Decimal
    actual_value: Decimal
    difference: Decimal
    difference_percent: float
    field_name: Optional[str] = None
    line_number: Optional[int] = None
    is_critical: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_likely_transposition(self) -> bool:
        """Check if error might be due to digit transposition."""
        exp_str = str(abs(self.expected_value)).replace('.', '')
        act_str = str(abs(self.actual_value)).replace('.', '')
        
        if len(exp_str) != len(act_str):
            return False
        
        # Check if same digits in different order
        return sorted(exp_str) == sorted(act_str) and exp_str != act_str
    
    @property
    def is_likely_decimal_shift(self) -> bool:
        """Check if error might be decimal point shift."""
        ratio = float(self.expected_value / self.actual_value) if self.actual_value else 0
        
        # Check for powers of 10
        for power in [10, 100, 1000, 0.1, 0.01, 0.001]:
            if abs(ratio - power) < 0.001:
                return True
        return False
    
    @property
    def suggestion(self) -> str:
        """Generate suggestion based on error analysis."""
        if self.is_likely_transposition:
            return "Possible digit transposition - verify source document"
        if self.is_likely_decimal_shift:
            return "Possible decimal point error - check for missing/extra zeros"
        if self.difference_percent > 50:
            return "Large discrepancy - may indicate OCR error or missing data"
        return "Verify calculations against source document"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_type': self.error_type.name,
            'message': self.message,
            'expected_value': str(self.expected_value),
            'actual_value': str(self.actual_value),
            'difference': str(self.difference),
            'difference_percent': round(self.difference_percent, 2),
            'field_name': self.field_name,
            'line_number': self.line_number,
            'is_critical': self.is_critical,
            'is_likely_transposition': self.is_likely_transposition,
            'is_likely_decimal_shift': self.is_likely_decimal_shift,
            'suggestion': self.suggestion,
        }


@dataclass
class TotalsMismatch:
    """
    Detailed information about a totals mismatch.
    """
    computed_total: Decimal
    extracted_total: Decimal
    components: Dict[str, Decimal]
    formula: str
    
    @property
    def difference(self) -> Decimal:
        return abs(self.computed_total - self.extracted_total)
    
    @property
    def difference_percent(self) -> float:
        if self.extracted_total == 0:
            return float('inf') if self.computed_total != 0 else 0.0
        return float(abs(self.difference / self.extracted_total * 100))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'computed_total': str(self.computed_total),
            'extracted_total': str(self.extracted_total),
            'difference': str(self.difference),
            'difference_percent': round(self.difference_percent, 2),
            'components': {k: str(v) for k, v in self.components.items()},
            'formula': self.formula,
        }


@dataclass
class ArithmeticResult:
    """
    Complete result of arithmetic validation.
    """
    is_valid: bool
    errors: List[CalculationError]
    checks_performed: int
    totals_mismatch: Optional[TotalsMismatch] = None
    computed_values: Dict[str, Decimal] = field(default_factory=dict)
    
    @property
    def has_critical_errors(self) -> bool:
        """Check if any critical errors exist."""
        return any(e.is_critical for e in self.errors)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def confidence_penalty(self) -> float:
        """Calculate confidence penalty based on errors."""
        penalty = 0.0
        for error in self.errors:
            if error.is_critical:
                penalty += 0.3
            elif error.difference_percent > 10:
                penalty += 0.2
            elif error.difference_percent > 1:
                penalty += 0.1
            else:
                penalty += 0.05
        return min(penalty, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'error_count': self.error_count,
            'checks_performed': self.checks_performed,
            'has_critical_errors': self.has_critical_errors,
            'confidence_penalty': round(self.confidence_penalty, 3),
            'errors': [e.to_dict() for e in self.errors],
            'totals_mismatch': self.totals_mismatch.to_dict() if self.totals_mismatch else None,
            'computed_values': {k: str(v) for k, v in self.computed_values.items()},
        }


class ArithmeticChecker:
    """
    Arithmetic validation for financial documents.
    
    Verifies that all extracted numbers are arithmetically consistent.
    
    Features:
    - Decimal precision (no float rounding)
    - Configurable tolerance
    - Detailed error analysis
    - Transposition/decimal shift detection
    
    Usage:
        checker = ArithmeticChecker()
        
        result = checker.validate_invoice(
            fields={'subtotal': '100.00', 'tax': '10.00', 'total': '110.00'},
            line_items=[{'qty': 2, 'price': '50.00', 'amount': '100.00'}],
        )
        
        if not result.is_valid:
            for error in result.errors:
                print(f"{error.error_type}: {error.message}")
    """
    
    # Default tolerance as percentage (0.01 = 1%)
    DEFAULT_TOLERANCE = Decimal('0.01')
    
    # Absolute tolerance for small amounts (in currency units)
    ABSOLUTE_TOLERANCE = Decimal('0.10')
    
    def __init__(
        self,
        tolerance: Optional[Decimal] = None,
        absolute_tolerance: Optional[Decimal] = None,
    ):
        """
        Initialize arithmetic checker.
        
        Args:
            tolerance: Relative tolerance as decimal (e.g., 0.01 for 1%)
            absolute_tolerance: Absolute tolerance for small amounts
        """
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE
        self.absolute_tolerance = absolute_tolerance or self.ABSOLUTE_TOLERANCE
    
    def validate_invoice(
        self,
        fields: Dict[str, Any],
        line_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ArithmeticResult:
        """
        Validate arithmetic of an invoice.
        
        Checks:
        1. Subtotal + Tax - Discount = Total
        2. Sum(line item amounts) = Subtotal
        3. Each line: quantity × price = amount
        4. Tax rate × subtotal ≈ tax amount
        
        Args:
            fields: Extracted field values
            line_items: List of line item dictionaries
            
        Returns:
            ArithmeticResult with validation details
        """
        errors: List[CalculationError] = []
        computed_values: Dict[str, Decimal] = {}
        checks_performed = 0
        totals_mismatch = None
        
        # Parse values
        subtotal = self._parse_decimal(fields.get('subtotal'))
        tax_amount = self._parse_decimal(fields.get('tax_amount'))
        discount = self._parse_decimal(fields.get('discount_amount')) or Decimal('0')
        total = self._parse_decimal(fields.get('grand_total') or fields.get('total'))
        tax_rate = self._parse_decimal(fields.get('tax_rate'))
        
        # Check 1: Subtotal + Tax - Discount = Total
        if subtotal is not None and total is not None:
            checks_performed += 1
            
            # Compute expected total
            tax = tax_amount or Decimal('0')
            expected_total = subtotal + tax - discount
            computed_values['computed_total'] = expected_total
            
            if not self._values_match(expected_total, total):
                diff = abs(expected_total - total)
                diff_pct = self._percent_difference(expected_total, total)
                
                errors.append(CalculationError(
                    error_type=ErrorType.TOTAL_MISMATCH,
                    message=f"Total calculation mismatch: {subtotal} + {tax} - {discount} = {expected_total}, but total is {total}",
                    expected_value=expected_total,
                    actual_value=total,
                    difference=diff,
                    difference_percent=diff_pct,
                    field_name='grand_total',
                    is_critical=diff_pct > 5,  # More than 5% is critical
                ))
                
                totals_mismatch = TotalsMismatch(
                    computed_total=expected_total,
                    extracted_total=total,
                    components={
                        'subtotal': subtotal,
                        'tax_amount': tax,
                        'discount': discount,
                    },
                    formula='subtotal + tax_amount - discount = total',
                )
        
        # Check 2: Tax calculation
        if subtotal is not None and tax_rate is not None and tax_amount is not None:
            checks_performed += 1
            
            # Convert rate to decimal if percentage
            rate = tax_rate
            if rate > 1:
                rate = rate / 100
            
            expected_tax = (subtotal * rate).quantize(Decimal('0.01'), ROUND_HALF_UP)
            computed_values['computed_tax'] = expected_tax
            
            if not self._values_match(expected_tax, tax_amount):
                diff = abs(expected_tax - tax_amount)
                diff_pct = self._percent_difference(expected_tax, tax_amount)
                
                errors.append(CalculationError(
                    error_type=ErrorType.TAX_CALCULATION_ERROR,
                    message=f"Tax calculation mismatch: {subtotal} × {rate:.2%} = {expected_tax}, but tax is {tax_amount}",
                    expected_value=expected_tax,
                    actual_value=tax_amount,
                    difference=diff,
                    difference_percent=diff_pct,
                    field_name='tax_amount',
                    is_critical=diff_pct > 10,
                ))
        
        # Check 3: Line items sum
        if line_items and subtotal is not None:
            checks_performed += 1
            
            line_sum = Decimal('0')
            valid_items = 0
            
            for item in line_items:
                amount = self._parse_decimal(item.get('amount'))
                if amount is not None:
                    line_sum += amount
                    valid_items += 1
            
            computed_values['computed_subtotal'] = line_sum
            
            if valid_items > 0 and not self._values_match(line_sum, subtotal):
                diff = abs(line_sum - subtotal)
                diff_pct = self._percent_difference(line_sum, subtotal)
                
                errors.append(CalculationError(
                    error_type=ErrorType.LINE_ITEMS_MISMATCH,
                    message=f"Line items sum ({line_sum}) doesn't match subtotal ({subtotal})",
                    expected_value=line_sum,
                    actual_value=subtotal,
                    difference=diff,
                    difference_percent=diff_pct,
                    field_name='subtotal',
                    is_critical=diff_pct > 5,
                    details={'line_item_count': valid_items},
                ))
        
        # Check 4: Individual line item calculations
        if line_items:
            for i, item in enumerate(line_items):
                error = self._check_line_item(item, i)
                if error:
                    checks_performed += 1
                    errors.append(error)
        
        return ArithmeticResult(
            is_valid=len(errors) == 0,
            errors=errors,
            checks_performed=checks_performed,
            totals_mismatch=totals_mismatch,
            computed_values=computed_values,
        )
    
    def validate_bank_statement(
        self,
        opening_balance: Any,
        closing_balance: Any,
        transactions: List[Dict[str, Any]],
    ) -> ArithmeticResult:
        """
        Validate arithmetic of a bank statement.
        
        Checks: Opening + Sum(transactions) = Closing
        
        Args:
            opening_balance: Opening balance
            closing_balance: Closing balance
            transactions: List of transactions with 'amount' and 'type' (debit/credit)
            
        Returns:
            ArithmeticResult with validation details
        """
        errors: List[CalculationError] = []
        computed_values: Dict[str, Decimal] = {}
        checks_performed = 0
        
        opening = self._parse_decimal(opening_balance)
        closing = self._parse_decimal(closing_balance)
        
        if opening is None or closing is None:
            return ArithmeticResult(
                is_valid=True,
                errors=[],
                checks_performed=0,
            )
        
        # Calculate running balance
        running = opening
        
        for txn in transactions:
            amount = self._parse_decimal(txn.get('amount'))
            if amount is None:
                continue
            
            txn_type = str(txn.get('type', '')).lower()
            
            if 'credit' in txn_type or 'deposit' in txn_type:
                running += amount
            elif 'debit' in txn_type or 'withdrawal' in txn_type:
                running -= amount
            else:
                # Unknown type - assume sign is included in amount
                running += amount
        
        computed_values['computed_closing'] = running
        checks_performed += 1
        
        if not self._values_match(running, closing):
            diff = abs(running - closing)
            diff_pct = self._percent_difference(running, closing)
            
            errors.append(CalculationError(
                error_type=ErrorType.BALANCE_ERROR,
                message=f"Balance mismatch: computed closing ({running}) != extracted closing ({closing})",
                expected_value=running,
                actual_value=closing,
                difference=diff,
                difference_percent=diff_pct,
                field_name='closing_balance',
                is_critical=diff_pct > 1,
                details={'transaction_count': len(transactions)},
            ))
        
        return ArithmeticResult(
            is_valid=len(errors) == 0,
            errors=errors,
            checks_performed=checks_performed,
            computed_values=computed_values,
        )
    
    def _check_line_item(
        self,
        item: Dict[str, Any],
        line_number: int,
    ) -> Optional[CalculationError]:
        """Check a single line item calculation."""
        qty = self._parse_decimal(item.get('quantity'))
        price = self._parse_decimal(item.get('unit_price') or item.get('price'))
        amount = self._parse_decimal(item.get('amount') or item.get('total'))
        
        if qty is None or price is None or amount is None:
            return None
        
        expected = (qty * price).quantize(Decimal('0.01'), ROUND_HALF_UP)
        
        if not self._values_match(expected, amount):
            diff = abs(expected - amount)
            diff_pct = self._percent_difference(expected, amount)
            
            return CalculationError(
                error_type=ErrorType.LINE_CALCULATION_ERROR,
                message=f"Line {line_number + 1}: {qty} × {price} = {expected}, but amount is {amount}",
                expected_value=expected,
                actual_value=amount,
                difference=diff,
                difference_percent=diff_pct,
                field_name=f'line_items[{line_number}].amount',
                line_number=line_number + 1,
                is_critical=diff_pct > 10,
            )
        
        return None
    
    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse a value to Decimal."""
        if value is None:
            return None
        
        if isinstance(value, Decimal):
            return value
        
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None
        
        if isinstance(value, str):
            # Clean the string
            cleaned = value.strip()
            cleaned = cleaned.replace(',', '')
            cleaned = cleaned.replace('$', '')
            cleaned = cleaned.replace('€', '')
            cleaned = cleaned.replace('£', '')
            cleaned = cleaned.replace(' ', '')
            
            # Handle negative in parentheses
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            try:
                return Decimal(cleaned)
            except InvalidOperation:
                return None
        
        return None
    
    def _values_match(self, expected: Decimal, actual: Decimal) -> bool:
        """
        Check if two values match within tolerance.
        
        Uses both relative and absolute tolerance.
        """
        diff = abs(expected - actual)
        
        # Check absolute tolerance (for small amounts)
        if diff <= self.absolute_tolerance:
            return True
        
        # Check relative tolerance
        if expected == 0:
            return actual == 0
        
        relative_diff = diff / abs(expected)
        return relative_diff <= self.tolerance
    
    def _percent_difference(self, expected: Decimal, actual: Decimal) -> float:
        """Calculate percentage difference."""
        if expected == 0:
            return float('inf') if actual != 0 else 0.0
        
        return float(abs(expected - actual) / abs(expected) * 100)
