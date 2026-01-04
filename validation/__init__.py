"""
Semantic Validation Package

This package provides business-logic validation for extracted document data.
It goes beyond format validation to check semantic correctness.

Validation Types:
- Arithmetic checks (totals, sums, calculations)
- Cross-field validation (dates, references)
- Business rule validation (tax rates, discounts)
- Consistency checks (line items vs totals)

Key Principle: Semantic errors are more dangerous than format errors.
A well-formatted but arithmetically incorrect total can cause real damage.

Usage:
    from validation import SemanticValidator, ArithmeticChecker
    
    validator = SemanticValidator()
    results = validator.validate(extracted_fields)
    
    for issue in results.issues:
        print(f"{issue.severity}: {issue.message}")
"""

from .semantic_rules import (
    SemanticValidator,
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    IssueCategory,
)
from .arithmetic_checks import (
    ArithmeticChecker,
    ArithmeticResult,
    CalculationError,
    TotalsMismatch,
)

__all__ = [
    # Semantic validation
    'SemanticValidator',
    'ValidationResult',
    'ValidationIssue',
    'IssueSeverity',
    'IssueCategory',
    
    # Arithmetic checks
    'ArithmeticChecker',
    'ArithmeticResult',
    'CalculationError',
    'TotalsMismatch',
]
