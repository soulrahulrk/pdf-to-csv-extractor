"""
Decision System Package

This package replaces simple confidence scores with an intelligent
decision system that provides actionable outcomes.

Confidence Score Problem:
- "0.75 confidence" is meaningless to users
- No clear threshold for accept/reject
- Doesn't explain WHY confidence is low
- Different fields need different thresholds

Decision System Solution:
- VERIFIED: High confidence, auto-accept
- LIKELY: Good confidence, may auto-accept
- REVIEW_REQUIRED: Needs human verification
- REJECTED: Cannot be trusted

Each decision includes:
- Reasons explaining the decision
- Specific issues found
- Suggested actions

Usage:
    from decision import DecisionEngine, Decision
    
    engine = DecisionEngine()
    result = engine.decide(
        field_name='invoice_number',
        value='INV-2024-001',
        confidence=0.85,
        extraction_method='text_layer',
    )
    
    if result.decision == Decision.REVIEW_REQUIRED:
        for reason in result.reasons:
            print(f"  - {reason}")
"""

from .decision_engine import (
    Decision,
    DecisionReason,
    FieldDecision,
    DocumentDecision,
    DecisionEngine,
    DecisionConfig,
)

__all__ = [
    'Decision',
    'DecisionReason',
    'FieldDecision',
    'DocumentDecision',
    'DecisionEngine',
    'DecisionConfig',
]
