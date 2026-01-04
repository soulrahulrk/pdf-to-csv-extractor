"""
Semantic Validation Rules

This module provides business-logic validation that checks for
semantic correctness, not just format correctness.

Why Semantic Validation:
- Format-correct data can still be semantically wrong
- "$100.00" is valid format but may be wrong amount
- Dates in correct format may be logically impossible
- Totals may not match line items
- Tax calculations may be incorrect

Validation Categories:
1. Temporal logic - date relationships make sense
2. Arithmetic integrity - calculations are correct
3. Reference consistency - IDs/numbers are valid
4. Business rules - domain-specific constraints
5. Completeness - required relationships exist

Design Philosophy:
- Flag issues, don't silently fix
- Provide actionable feedback
- Support review workflows
- Enable confidence scoring
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Union, Set

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    CRITICAL = auto()   # Document cannot be processed
    ERROR = auto()      # Likely incorrect, needs review
    WARNING = auto()    # Possibly incorrect, flag for review
    INFO = auto()       # Informational, may be correct


class IssueCategory(Enum):
    """Categories of validation issues."""
    ARITHMETIC = auto()        # Math doesn't add up
    TEMPORAL = auto()          # Date/time issues
    REFERENCE = auto()         # Invalid references
    BUSINESS_RULE = auto()     # Domain rule violation
    COMPLETENESS = auto()      # Missing required data
    CONSISTENCY = auto()       # Internal inconsistency
    PLAUSIBILITY = auto()      # Implausible values
    FORMAT = auto()            # Format issues


@dataclass
class ValidationIssue:
    """
    A single validation issue found during semantic validation.
    
    Provides detailed information about what's wrong and why,
    to support human review workflows.
    """
    severity: IssueSeverity
    category: IssueCategory
    message: str
    field_name: Optional[str] = None
    field_value: Any = None
    expected_value: Any = None
    rule_name: str = ''
    confidence_impact: float = 0.0  # How much to reduce confidence
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'severity': self.severity.name,
            'category': self.category.name,
            'message': self.message,
            'field_name': self.field_name,
            'field_value': str(self.field_value) if self.field_value else None,
            'expected_value': str(self.expected_value) if self.expected_value else None,
            'rule_name': self.rule_name,
            'confidence_impact': round(self.confidence_impact, 3),
            'suggestion': self.suggestion,
        }
    
    def __str__(self) -> str:
        parts = [f"[{self.severity.name}]"]
        if self.field_name:
            parts.append(f"{self.field_name}:")
        parts.append(self.message)
        return ' '.join(parts)


@dataclass
class ValidationResult:
    """
    Complete result of semantic validation.
    
    Contains all issues found and summary statistics.
    """
    is_valid: bool
    issues: List[ValidationIssue]
    fields_validated: int
    rules_applied: int
    confidence_adjustment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]
    
    @property
    def warning_issues(self) -> List[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]
    
    @property
    def has_critical(self) -> bool:
        """Check if there are critical issues."""
        return len(self.critical_issues) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return len(self.error_issues) > 0
    
    @property
    def issue_count(self) -> int:
        """Total number of issues."""
        return len(self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'issue_count': self.issue_count,
            'critical_count': len(self.critical_issues),
            'error_count': len(self.error_issues),
            'warning_count': len(self.warning_issues),
            'fields_validated': self.fields_validated,
            'rules_applied': self.rules_applied,
            'confidence_adjustment': round(self.confidence_adjustment, 3),
            'issues': [i.to_dict() for i in self.issues],
        }


class SemanticRule:
    """
    Base class for semantic validation rules.
    
    Subclass this to create custom validation rules.
    """
    
    name: str = 'unnamed_rule'
    category: IssueCategory = IssueCategory.BUSINESS_RULE
    required_fields: Set[str] = set()
    
    def __init__(self):
        self.enabled = True
    
    def applies_to(self, fields: Dict[str, Any]) -> bool:
        """Check if this rule applies to the given fields."""
        return all(f in fields and fields[f] is not None for f in self.required_fields)
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Validate fields and return any issues found.
        
        Override this in subclasses.
        """
        raise NotImplementedError


class DateOrderRule(SemanticRule):
    """Validates that date fields are in logical order."""
    
    name = 'date_order'
    category = IssueCategory.TEMPORAL
    
    def __init__(
        self,
        earlier_field: str,
        later_field: str,
        max_gap_days: Optional[int] = None,
    ):
        super().__init__()
        self.earlier_field = earlier_field
        self.later_field = later_field
        self.max_gap_days = max_gap_days
        self.required_fields = {earlier_field, later_field}
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        earlier = self._parse_date(fields.get(self.earlier_field))
        later = self._parse_date(fields.get(self.later_field))
        
        if earlier is None or later is None:
            return issues  # Can't validate if dates aren't parseable
        
        # Check order
        if later < earlier:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.TEMPORAL,
                message=f"{self.later_field} ({later}) is before {self.earlier_field} ({earlier})",
                field_name=self.later_field,
                field_value=later,
                expected_value=f"Date after {earlier}",
                rule_name=self.name,
                confidence_impact=0.3,
                suggestion=f"Check if dates are swapped",
            ))
        
        # Check gap if specified
        elif self.max_gap_days:
            gap = (later - earlier).days
            if gap > self.max_gap_days:
                issues.append(ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.TEMPORAL,
                    message=f"Gap of {gap} days between {self.earlier_field} and {self.later_field} exceeds {self.max_gap_days} days",
                    field_name=self.later_field,
                    rule_name=self.name,
                    confidence_impact=0.1,
                    details={'gap_days': gap},
                ))
        
        return issues
    
    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse date from various formats."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y']:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        return None


class ReasonableDateRule(SemanticRule):
    """Validates that dates are within reasonable bounds."""
    
    name = 'reasonable_date'
    category = IssueCategory.PLAUSIBILITY
    
    def __init__(
        self,
        field_name: str,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
        max_future_days: int = 365,
        max_past_years: int = 10,
    ):
        super().__init__()
        self.field_name = field_name
        self.min_date = min_date
        self.max_date = max_date
        self.max_future_days = max_future_days
        self.max_past_years = max_past_years
        self.required_fields = {field_name}
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        value = fields.get(self.field_name)
        
        parsed = self._parse_date(value)
        if parsed is None:
            return issues
        
        today = date.today()
        
        # Check future limit
        max_future = today + timedelta(days=self.max_future_days)
        if parsed > max_future:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.PLAUSIBILITY,
                message=f"{self.field_name} is too far in the future ({parsed})",
                field_name=self.field_name,
                field_value=parsed,
                rule_name=self.name,
                confidence_impact=0.2,
            ))
        
        # Check past limit
        min_past = today.replace(year=today.year - self.max_past_years)
        if parsed < min_past:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.PLAUSIBILITY,
                message=f"{self.field_name} is very old ({parsed})",
                field_name=self.field_name,
                field_value=parsed,
                rule_name=self.name,
                confidence_impact=0.15,
            ))
        
        # Check explicit bounds
        if self.min_date and parsed < self.min_date:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.TEMPORAL,
                message=f"{self.field_name} is before minimum date ({self.min_date})",
                field_name=self.field_name,
                field_value=parsed,
                expected_value=f">= {self.min_date}",
                rule_name=self.name,
                confidence_impact=0.25,
            ))
        
        if self.max_date and parsed > self.max_date:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.TEMPORAL,
                message=f"{self.field_name} is after maximum date ({self.max_date})",
                field_name=self.field_name,
                field_value=parsed,
                expected_value=f"<= {self.max_date}",
                rule_name=self.name,
                confidence_impact=0.25,
            ))
        
        return issues
    
    def _parse_date(self, value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                try:
                    return datetime.strptime(value, fmt).date()
                except ValueError:
                    continue
        return None


class TaxRatePlausibilityRule(SemanticRule):
    """Validates that tax rates are within plausible bounds."""
    
    name = 'tax_rate_plausibility'
    category = IssueCategory.PLAUSIBILITY
    required_fields = {'tax_rate'}
    
    def __init__(
        self,
        min_rate: float = 0.0,
        max_rate: float = 0.35,  # 35% is high for most jurisdictions
        common_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.common_rates = common_rates or [0.0, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.21, 0.23, 0.25]
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        rate = fields.get('tax_rate')
        
        if rate is None:
            return issues
        
        try:
            rate_value = float(rate)
        except (ValueError, TypeError):
            return issues
        
        # Convert percentage to decimal if needed
        if rate_value > 1:
            rate_value = rate_value / 100
        
        if rate_value < self.min_rate:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.PLAUSIBILITY,
                message=f"Tax rate {rate_value:.1%} is below minimum expected ({self.min_rate:.1%})",
                field_name='tax_rate',
                field_value=rate_value,
                rule_name=self.name,
                confidence_impact=0.1,
            ))
        
        if rate_value > self.max_rate:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.PLAUSIBILITY,
                message=f"Tax rate {rate_value:.1%} exceeds maximum expected ({self.max_rate:.1%})",
                field_name='tax_rate',
                field_value=rate_value,
                expected_value=f"<= {self.max_rate:.1%}",
                rule_name=self.name,
                confidence_impact=0.25,
                suggestion="Verify tax rate is correctly extracted",
            ))
        
        # Check if rate is close to a common rate
        if self.common_rates:
            closest = min(self.common_rates, key=lambda x: abs(x - rate_value))
            diff = abs(closest - rate_value)
            
            if 0.001 < diff < 0.02:  # Not exact but close
                issues.append(ValidationIssue(
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.PLAUSIBILITY,
                    message=f"Tax rate {rate_value:.1%} is close to common rate {closest:.1%}",
                    field_name='tax_rate',
                    field_value=rate_value,
                    expected_value=closest,
                    rule_name=self.name,
                    confidence_impact=0.05,
                    suggestion=f"Consider if rate should be {closest:.1%}",
                ))
        
        return issues


class AmountPlausibilityRule(SemanticRule):
    """Validates that monetary amounts are plausible."""
    
    name = 'amount_plausibility'
    category = IssueCategory.PLAUSIBILITY
    
    def __init__(
        self,
        field_name: str,
        min_amount: float = 0.0,
        max_amount: float = 10_000_000,  # 10 million
        must_be_positive: bool = True,
    ):
        super().__init__()
        self.field_name = field_name
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.must_be_positive = must_be_positive
        self.required_fields = {field_name}
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        value = fields.get(self.field_name)
        
        if value is None:
            return issues
        
        try:
            amount = float(str(value).replace(',', '').replace('$', '').replace('€', '').replace('£', ''))
        except (ValueError, TypeError):
            return issues
        
        if self.must_be_positive and amount < 0:
            issues.append(ValidationIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.PLAUSIBILITY,
                message=f"{self.field_name} is negative ({amount})",
                field_name=self.field_name,
                field_value=amount,
                expected_value=">= 0",
                rule_name=self.name,
                confidence_impact=0.3,
            ))
        
        if amount < self.min_amount:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.PLAUSIBILITY,
                message=f"{self.field_name} ({amount}) is below minimum ({self.min_amount})",
                field_name=self.field_name,
                field_value=amount,
                rule_name=self.name,
                confidence_impact=0.1,
            ))
        
        if amount > self.max_amount:
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.PLAUSIBILITY,
                message=f"{self.field_name} ({amount}) exceeds maximum ({self.max_amount})",
                field_name=self.field_name,
                field_value=amount,
                rule_name=self.name,
                confidence_impact=0.15,
            ))
        
        return issues


class ReferenceFormatRule(SemanticRule):
    """Validates reference number formats."""
    
    name = 'reference_format'
    category = IssueCategory.FORMAT
    
    def __init__(
        self,
        field_name: str,
        pattern: str,
        description: str = 'reference number',
    ):
        super().__init__()
        self.field_name = field_name
        self.pattern = re.compile(pattern)
        self.description = description
        self.required_fields = {field_name}
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        value = fields.get(self.field_name)
        
        if value is None:
            return issues
        
        value_str = str(value).strip()
        
        if not self.pattern.match(value_str):
            issues.append(ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.FORMAT,
                message=f"{self.field_name} doesn't match expected {self.description} format",
                field_name=self.field_name,
                field_value=value_str,
                rule_name=self.name,
                confidence_impact=0.1,
            ))
        
        return issues


class RequiredFieldsRule(SemanticRule):
    """Validates that required fields are present."""
    
    name = 'required_fields'
    category = IssueCategory.COMPLETENESS
    
    def __init__(
        self,
        required: List[str],
        severity: IssueSeverity = IssueSeverity.WARNING,
    ):
        super().__init__()
        self.required = required
        self.severity = severity
    
    def applies_to(self, fields: Dict[str, Any]) -> bool:
        return True  # Always applies
    
    def validate(self, fields: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        for field_name in self.required:
            value = fields.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                issues.append(ValidationIssue(
                    severity=self.severity,
                    category=IssueCategory.COMPLETENESS,
                    message=f"Required field '{field_name}' is missing or empty",
                    field_name=field_name,
                    rule_name=self.name,
                    confidence_impact=0.15,
                ))
        
        return issues


class SemanticValidator:
    """
    Main semantic validation engine.
    
    Applies a collection of semantic rules to validate extracted data.
    
    Usage:
        validator = SemanticValidator()
        
        # Add custom rules
        validator.add_rule(DateOrderRule('invoice_date', 'due_date'))
        
        # Validate
        result = validator.validate(extracted_fields)
        
        if not result.is_valid:
            for issue in result.issues:
                print(issue)
    """
    
    def __init__(self, document_type: str = 'invoice'):
        """
        Initialize validator.
        
        Args:
            document_type: Type of document ('invoice', 'receipt', etc.)
        """
        self.document_type = document_type
        self.rules: List[SemanticRule] = []
        
        # Add default rules based on document type
        self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default rules for the document type."""
        if self.document_type == 'invoice':
            self._add_invoice_rules()
        elif self.document_type == 'receipt':
            self._add_receipt_rules()
        elif self.document_type == 'bank_statement':
            self._add_bank_statement_rules()
    
    def _add_invoice_rules(self):
        """Add invoice-specific rules."""
        # Date rules
        self.add_rule(DateOrderRule('invoice_date', 'due_date', max_gap_days=365))
        self.add_rule(ReasonableDateRule('invoice_date'))
        self.add_rule(ReasonableDateRule('due_date'))
        
        # Tax rate rule
        self.add_rule(TaxRatePlausibilityRule())
        
        # Amount rules
        self.add_rule(AmountPlausibilityRule('subtotal'))
        self.add_rule(AmountPlausibilityRule('tax_amount'))
        self.add_rule(AmountPlausibilityRule('grand_total'))
        self.add_rule(AmountPlausibilityRule('discount_amount', min_amount=0.0, max_amount=100_000))
        
        # Reference format
        self.add_rule(ReferenceFormatRule(
            'invoice_number',
            r'^[A-Za-z0-9\-_/]+$',
            'invoice number (alphanumeric with dashes)',
        ))
        
        # Required fields
        self.add_rule(RequiredFieldsRule(['invoice_number', 'invoice_date', 'grand_total']))
    
    def _add_receipt_rules(self):
        """Add receipt-specific rules."""
        self.add_rule(ReasonableDateRule('date'))
        self.add_rule(AmountPlausibilityRule('total', max_amount=50_000))
        self.add_rule(TaxRatePlausibilityRule())
        self.add_rule(RequiredFieldsRule(['date', 'total']))
    
    def _add_bank_statement_rules(self):
        """Add bank statement rules."""
        self.add_rule(DateOrderRule('statement_start_date', 'statement_end_date', max_gap_days=31))
        self.add_rule(ReasonableDateRule('statement_start_date'))
        self.add_rule(ReasonableDateRule('statement_end_date'))
        self.add_rule(AmountPlausibilityRule('opening_balance', must_be_positive=False))
        self.add_rule(AmountPlausibilityRule('closing_balance', must_be_positive=False))
    
    def add_rule(self, rule: SemanticRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        return len(self.rules) < original_count
    
    def validate(
        self,
        fields: Dict[str, Any],
        line_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        Validate extracted fields.
        
        Args:
            fields: Dictionary of extracted field values
            line_items: Optional list of line item dictionaries
            
        Returns:
            ValidationResult with all issues found
        """
        issues: List[ValidationIssue] = []
        rules_applied = 0
        
        # Apply each rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if not rule.applies_to(fields):
                continue
            
            rules_applied += 1
            
            try:
                rule_issues = rule.validate(fields)
                issues.extend(rule_issues)
            except Exception as e:
                logger.warning(f"Rule {rule.name} failed: {e}")
        
        # Validate line items if provided
        if line_items:
            line_item_issues = self._validate_line_items(line_items, fields)
            issues.extend(line_item_issues)
        
        # Calculate total confidence impact
        confidence_adjustment = sum(i.confidence_impact for i in issues)
        
        # Determine overall validity
        is_valid = not any(
            i.severity in (IssueSeverity.CRITICAL, IssueSeverity.ERROR)
            for i in issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            fields_validated=len(fields),
            rules_applied=rules_applied,
            confidence_adjustment=-confidence_adjustment,
            metadata={
                'document_type': self.document_type,
                'line_items_validated': len(line_items) if line_items else 0,
            }
        )
    
    def _validate_line_items(
        self,
        line_items: List[Dict[str, Any]],
        fields: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Validate line items and their relationship to totals."""
        issues = []
        
        if not line_items:
            return issues
        
        # Check each line item
        for i, item in enumerate(line_items):
            # Quantity × Unit Price = Amount check
            qty = self._parse_number(item.get('quantity'))
            price = self._parse_number(item.get('unit_price'))
            amount = self._parse_number(item.get('amount'))
            
            if qty is not None and price is not None and amount is not None:
                expected = qty * price
                if not self._amounts_match(expected, amount):
                    issues.append(ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.ARITHMETIC,
                        message=f"Line item {i+1}: quantity × price ({qty} × {price} = {expected:.2f}) doesn't match amount ({amount})",
                        field_name=f'line_items[{i}].amount',
                        field_value=amount,
                        expected_value=expected,
                        rule_name='line_item_calculation',
                        confidence_impact=0.2,
                    ))
        
        # Sum of line items vs subtotal
        line_total = sum(
            self._parse_number(item.get('amount')) or 0
            for item in line_items
        )
        
        subtotal = self._parse_number(fields.get('subtotal'))
        if subtotal is not None and line_total > 0:
            if not self._amounts_match(line_total, subtotal, tolerance=0.05):
                issues.append(ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    category=IssueCategory.ARITHMETIC,
                    message=f"Sum of line items ({line_total:.2f}) doesn't match subtotal ({subtotal})",
                    field_name='subtotal',
                    field_value=subtotal,
                    expected_value=line_total,
                    rule_name='line_items_sum',
                    confidence_impact=0.3,
                    suggestion="Review line items and subtotal for errors",
                ))
        
        return issues
    
    def _parse_number(self, value: Any) -> Optional[float]:
        """Parse a number from various formats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                cleaned = value.replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
                return float(cleaned)
            except ValueError:
                return None
        return None
    
    def _amounts_match(
        self,
        a: float,
        b: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Check if two amounts match within tolerance."""
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return abs(a - b) < tolerance
        return abs(a - b) / max(abs(a), abs(b)) < tolerance
