"""
Decision Engine

This module provides intelligent decision-making for document extraction.
It replaces numeric confidence scores with actionable decisions.

Decision Levels:
- VERIFIED: Multiple signals confirm correctness (>95% confidence)
- LIKELY: Strong signals, acceptable for most use cases (75-95%)
- REVIEW_REQUIRED: Ambiguous signals, needs human review (50-75%)
- REJECTED: Too many issues, cannot be trusted (<50%)

Decision Factors:
- Raw extraction confidence
- Extraction method (text layer vs OCR)
- Spatial alignment quality
- Validation rule results
- Cross-field consistency
- Business rule compliance

Design Goals:
- No magic thresholds
- Explainable decisions
- Configurable per field type
- Support review workflows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Decision outcomes for extracted data."""
    
    VERIFIED = auto()        # High confidence, auto-accept
    LIKELY = auto()          # Good confidence, may auto-accept
    REVIEW_REQUIRED = auto() # Needs human verification
    REJECTED = auto()        # Cannot be trusted
    
    @property
    def is_acceptable(self) -> bool:
        """Check if this decision allows automated processing."""
        return self in (Decision.VERIFIED, Decision.LIKELY)
    
    @property
    def needs_review(self) -> bool:
        """Check if this decision requires human review."""
        return self in (Decision.REVIEW_REQUIRED, Decision.REJECTED)
    
    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            Decision.VERIFIED: "✓ Verified",
            Decision.LIKELY: "○ Likely",
            Decision.REVIEW_REQUIRED: "⚠ Review Required",
            Decision.REJECTED: "✗ Rejected",
        }
        return names.get(self, self.name)


class DecisionReason(Enum):
    """Reasons that affect decisions."""
    
    # Positive signals
    TEXT_LAYER_EXTRACTION = auto()    # Extracted from text layer (reliable)
    PATTERN_MATCH = auto()            # Matched expected pattern
    SPATIAL_ALIGNMENT = auto()        # Good spatial positioning
    CROSS_VALIDATED = auto()          # Confirmed by related fields
    ARITHMETIC_VERIFIED = auto()      # Math checks passed
    BUSINESS_RULE_PASSED = auto()     # Domain rules satisfied
    HIGH_OCR_CONFIDENCE = auto()      # OCR very confident
    
    # Negative signals
    OCR_EXTRACTED = auto()            # From OCR (less reliable)
    LOW_CONFIDENCE = auto()           # Extraction confidence low
    PATTERN_MISMATCH = auto()         # Doesn't match expected format
    SPATIAL_WEAK = auto()             # Poor spatial alignment
    VALIDATION_FAILED = auto()        # Failed validation rules
    ARITHMETIC_ERROR = auto()         # Math doesn't check out
    BUSINESS_RULE_VIOLATED = auto()   # Domain rule violation
    MULTIPLE_CANDIDATES = auto()      # Ambiguous extraction
    OCR_QUALITY_LOW = auto()          # Poor OCR quality
    MISSING_CONTEXT = auto()          # No supporting context
    IMPLAUSIBLE_VALUE = auto()        # Value seems wrong
    
    @property
    def is_positive(self) -> bool:
        """Whether this reason increases confidence."""
        positive = {
            DecisionReason.TEXT_LAYER_EXTRACTION,
            DecisionReason.PATTERN_MATCH,
            DecisionReason.SPATIAL_ALIGNMENT,
            DecisionReason.CROSS_VALIDATED,
            DecisionReason.ARITHMETIC_VERIFIED,
            DecisionReason.BUSINESS_RULE_PASSED,
            DecisionReason.HIGH_OCR_CONFIDENCE,
        }
        return self in positive
    
    @property
    def weight(self) -> float:
        """How much this reason affects the decision."""
        weights = {
            # Positive
            DecisionReason.TEXT_LAYER_EXTRACTION: 0.15,
            DecisionReason.PATTERN_MATCH: 0.10,
            DecisionReason.SPATIAL_ALIGNMENT: 0.10,
            DecisionReason.CROSS_VALIDATED: 0.15,
            DecisionReason.ARITHMETIC_VERIFIED: 0.20,
            DecisionReason.BUSINESS_RULE_PASSED: 0.10,
            DecisionReason.HIGH_OCR_CONFIDENCE: 0.05,
            
            # Negative
            DecisionReason.OCR_EXTRACTED: -0.10,
            DecisionReason.LOW_CONFIDENCE: -0.15,
            DecisionReason.PATTERN_MISMATCH: -0.15,
            DecisionReason.SPATIAL_WEAK: -0.10,
            DecisionReason.VALIDATION_FAILED: -0.20,
            DecisionReason.ARITHMETIC_ERROR: -0.30,
            DecisionReason.BUSINESS_RULE_VIOLATED: -0.20,
            DecisionReason.MULTIPLE_CANDIDATES: -0.15,
            DecisionReason.OCR_QUALITY_LOW: -0.20,
            DecisionReason.MISSING_CONTEXT: -0.10,
            DecisionReason.IMPLAUSIBLE_VALUE: -0.25,
        }
        return weights.get(self, 0.0)
    
    @property
    def display_message(self) -> str:
        """Human-readable description."""
        messages = {
            DecisionReason.TEXT_LAYER_EXTRACTION: "Extracted from PDF text layer (reliable)",
            DecisionReason.PATTERN_MATCH: "Matches expected format pattern",
            DecisionReason.SPATIAL_ALIGNMENT: "Good spatial alignment with label",
            DecisionReason.CROSS_VALIDATED: "Confirmed by related fields",
            DecisionReason.ARITHMETIC_VERIFIED: "Arithmetic verification passed",
            DecisionReason.BUSINESS_RULE_PASSED: "Business rules satisfied",
            DecisionReason.HIGH_OCR_CONFIDENCE: "High OCR confidence score",
            
            DecisionReason.OCR_EXTRACTED: "Extracted via OCR (may contain errors)",
            DecisionReason.LOW_CONFIDENCE: "Low extraction confidence",
            DecisionReason.PATTERN_MISMATCH: "Doesn't match expected format",
            DecisionReason.SPATIAL_WEAK: "Weak spatial alignment with context",
            DecisionReason.VALIDATION_FAILED: "Failed validation checks",
            DecisionReason.ARITHMETIC_ERROR: "Arithmetic verification failed",
            DecisionReason.BUSINESS_RULE_VIOLATED: "Violates business rules",
            DecisionReason.MULTIPLE_CANDIDATES: "Multiple possible values found",
            DecisionReason.OCR_QUALITY_LOW: "Poor OCR quality detected",
            DecisionReason.MISSING_CONTEXT: "No supporting context found",
            DecisionReason.IMPLAUSIBLE_VALUE: "Value appears implausible",
        }
        return messages.get(self, self.name)


@dataclass
class FieldDecision:
    """
    Decision for a single extracted field.
    
    Contains the decision, supporting reasons, and suggested actions.
    """
    field_name: str
    value: Any
    decision: Decision
    reasons: List[DecisionReason]
    confidence_score: float  # Original confidence (for reference)
    adjusted_score: float    # After applying reasons
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def positive_reasons(self) -> List[DecisionReason]:
        """Get positive contributing reasons."""
        return [r for r in self.reasons if r.is_positive]
    
    @property
    def negative_reasons(self) -> List[DecisionReason]:
        """Get negative contributing reasons."""
        return [r for r in self.reasons if not r.is_positive]
    
    @property
    def is_acceptable(self) -> bool:
        """Whether this field can be auto-accepted."""
        return self.decision.is_acceptable
    
    @property
    def explanation(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Field: {self.field_name}",
            f"Value: {self.value}",
            f"Decision: {self.decision.display_name}",
            "",
            "Reasons:",
        ]
        
        for reason in self.positive_reasons:
            lines.append(f"  + {reason.display_message}")
        
        for reason in self.negative_reasons:
            lines.append(f"  - {reason.display_message}")
        
        if self.suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'field_name': self.field_name,
            'value': str(self.value) if self.value is not None else None,
            'decision': self.decision.name,
            'reasons': [r.name for r in self.reasons],
            'reason_details': [
                {'name': r.name, 'message': r.display_message, 'positive': r.is_positive}
                for r in self.reasons
            ],
            'confidence_score': round(self.confidence_score, 3),
            'adjusted_score': round(self.adjusted_score, 3),
            'suggestions': self.suggestions,
            'is_acceptable': self.is_acceptable,
        }


@dataclass
class DocumentDecision:
    """
    Decision for an entire document.
    
    Aggregates field-level decisions and provides document-level assessment.
    """
    document_id: str
    source_file: str
    field_decisions: Dict[str, FieldDecision]
    overall_decision: Decision
    extraction_complete: bool
    warnings: List[str]
    errors: List[str]
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def verified_count(self) -> int:
        """Number of verified fields."""
        return sum(1 for d in self.field_decisions.values() if d.decision == Decision.VERIFIED)
    
    @property
    def likely_count(self) -> int:
        """Number of likely fields."""
        return sum(1 for d in self.field_decisions.values() if d.decision == Decision.LIKELY)
    
    @property
    def review_count(self) -> int:
        """Number of fields needing review."""
        return sum(1 for d in self.field_decisions.values() if d.decision == Decision.REVIEW_REQUIRED)
    
    @property
    def rejected_count(self) -> int:
        """Number of rejected fields."""
        return sum(1 for d in self.field_decisions.values() if d.decision == Decision.REJECTED)
    
    @property
    def total_fields(self) -> int:
        """Total number of fields."""
        return len(self.field_decisions)
    
    @property
    def acceptance_rate(self) -> float:
        """Percentage of acceptable fields."""
        if self.total_fields == 0:
            return 0.0
        acceptable = self.verified_count + self.likely_count
        return acceptable / self.total_fields
    
    @property
    def needs_review(self) -> bool:
        """Whether document needs human review."""
        return self.overall_decision.needs_review
    
    @property
    def fields_needing_review(self) -> List[str]:
        """List of field names that need review."""
        return [
            name for name, d in self.field_decisions.items()
            if d.decision.needs_review
        ]
    
    def get_decision(self, field_name: str) -> Optional[FieldDecision]:
        """Get decision for a specific field."""
        return self.field_decisions.get(field_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'source_file': self.source_file,
            'overall_decision': self.overall_decision.name,
            'extraction_complete': self.extraction_complete,
            'statistics': {
                'total_fields': self.total_fields,
                'verified': self.verified_count,
                'likely': self.likely_count,
                'review_required': self.review_count,
                'rejected': self.rejected_count,
                'acceptance_rate': round(self.acceptance_rate, 2),
            },
            'fields_needing_review': self.fields_needing_review,
            'field_decisions': {
                name: d.to_dict() for name, d in self.field_decisions.items()
            },
            'warnings': self.warnings,
            'errors': self.errors,
        }


@dataclass
class DecisionConfig:
    """
    Configuration for decision engine.
    
    Allows customization of thresholds and rules per field type.
    """
    # Global thresholds
    verified_threshold: float = 0.90
    likely_threshold: float = 0.70
    review_threshold: float = 0.50
    
    # Field-specific thresholds (field_name -> threshold overrides)
    field_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Critical fields that must be verified for doc to pass
    critical_fields: Set[str] = field(default_factory=lambda: {
        'invoice_number', 'grand_total', 'invoice_date'
    })
    
    # Fields where OCR is particularly unreliable
    ocr_sensitive_fields: Set[str] = field(default_factory=lambda: {
        'grand_total', 'subtotal', 'tax_amount', 'invoice_number'
    })
    
    # Minimum acceptable for document approval
    min_acceptance_rate: float = 0.70
    
    def get_thresholds(self, field_name: str) -> Dict[str, float]:
        """Get thresholds for a specific field."""
        base = {
            'verified': self.verified_threshold,
            'likely': self.likely_threshold,
            'review': self.review_threshold,
        }
        
        if field_name in self.field_thresholds:
            base.update(self.field_thresholds[field_name])
        
        return base


class DecisionEngine:
    """
    Decision engine for document extraction.
    
    Converts confidence scores and validation results into
    actionable decisions with explanations.
    
    Usage:
        engine = DecisionEngine()
        
        # Make decision for a field
        field_decision = engine.decide_field(
            field_name='invoice_number',
            value='INV-2024-001',
            confidence=0.85,
            extraction_method='text_layer',
            validation_passed=True,
        )
        
        # Make decision for entire document
        doc_decision = engine.decide_document(
            fields={'invoice_number': 'INV-001', 'total': '100.00'},
            field_decisions={'invoice_number': field_decision, ...},
            warnings=['Some warning'],
            errors=[],
        )
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        """
        Initialize decision engine.
        
        Args:
            config: Decision configuration
        """
        self.config = config or DecisionConfig()
    
    def decide_field(
        self,
        field_name: str,
        value: Any,
        confidence: float,
        extraction_method: str = 'unknown',
        validation_passed: bool = True,
        arithmetic_passed: bool = True,
        spatial_score: float = 0.5,
        pattern_matched: bool = True,
        has_multiple_candidates: bool = False,
        ocr_quality: float = 0.8,
        additional_reasons: Optional[List[DecisionReason]] = None,
    ) -> FieldDecision:
        """
        Make a decision for a single field.
        
        Args:
            field_name: Name of the field
            value: Extracted value
            confidence: Raw extraction confidence (0-1)
            extraction_method: How the field was extracted
            validation_passed: Whether validation rules passed
            arithmetic_passed: Whether arithmetic checks passed
            spatial_score: Quality of spatial alignment (0-1)
            pattern_matched: Whether value matches expected pattern
            has_multiple_candidates: If there were ambiguous matches
            ocr_quality: OCR quality score if applicable
            additional_reasons: Extra reasons to consider
            
        Returns:
            FieldDecision with decision and explanation
        """
        reasons: List[DecisionReason] = additional_reasons or []
        suggestions: List[str] = []
        
        # Determine extraction method reasons
        if extraction_method in ('text_layer', 'pdfplumber', 'pymupdf'):
            reasons.append(DecisionReason.TEXT_LAYER_EXTRACTION)
        elif extraction_method in ('ocr', 'tesseract'):
            reasons.append(DecisionReason.OCR_EXTRACTED)
            
            if field_name in self.config.ocr_sensitive_fields:
                suggestions.append(f"Verify {field_name} manually - OCR extraction of numbers/IDs is error-prone")
            
            if ocr_quality >= 0.9:
                reasons.append(DecisionReason.HIGH_OCR_CONFIDENCE)
            elif ocr_quality < 0.6:
                reasons.append(DecisionReason.OCR_QUALITY_LOW)
                suggestions.append("Consider re-scanning at higher resolution")
        
        # Pattern matching
        if pattern_matched:
            reasons.append(DecisionReason.PATTERN_MATCH)
        else:
            reasons.append(DecisionReason.PATTERN_MISMATCH)
            suggestions.append(f"Check if {field_name} format is correct")
        
        # Spatial alignment
        if spatial_score >= 0.7:
            reasons.append(DecisionReason.SPATIAL_ALIGNMENT)
        elif spatial_score < 0.4:
            reasons.append(DecisionReason.SPATIAL_WEAK)
            suggestions.append("Verify field was extracted from correct location")
        
        # Validation
        if validation_passed:
            reasons.append(DecisionReason.BUSINESS_RULE_PASSED)
        else:
            reasons.append(DecisionReason.VALIDATION_FAILED)
            suggestions.append("Review validation errors")
        
        # Arithmetic
        if arithmetic_passed:
            reasons.append(DecisionReason.ARITHMETIC_VERIFIED)
        else:
            reasons.append(DecisionReason.ARITHMETIC_ERROR)
            suggestions.append("Verify calculations match source document")
        
        # Ambiguity
        if has_multiple_candidates:
            reasons.append(DecisionReason.MULTIPLE_CANDIDATES)
            suggestions.append("Multiple values found - verify correct one was selected")
        
        # Low base confidence
        if confidence < 0.5:
            reasons.append(DecisionReason.LOW_CONFIDENCE)
        
        # Check for implausible values
        if value is None or (isinstance(value, str) and not value.strip()):
            reasons.append(DecisionReason.MISSING_CONTEXT)
        
        # Calculate adjusted score
        adjusted_score = self._calculate_adjusted_score(confidence, reasons)
        
        # Make decision
        thresholds = self.config.get_thresholds(field_name)
        decision = self._score_to_decision(adjusted_score, thresholds)
        
        return FieldDecision(
            field_name=field_name,
            value=value,
            decision=decision,
            reasons=reasons,
            confidence_score=confidence,
            adjusted_score=adjusted_score,
            suggestions=suggestions,
        )
    
    def decide_document(
        self,
        document_id: str,
        source_file: str,
        field_decisions: Dict[str, FieldDecision],
        warnings: List[str],
        errors: List[str],
        extraction_metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentDecision:
        """
        Make overall decision for a document.
        
        Args:
            document_id: Unique document identifier
            source_file: Source file path/name
            field_decisions: Per-field decisions
            warnings: Extraction warnings
            errors: Extraction errors
            extraction_metadata: Additional metadata
            
        Returns:
            DocumentDecision with overall assessment
        """
        # Check critical fields
        critical_ok = True
        for field_name in self.config.critical_fields:
            if field_name in field_decisions:
                if not field_decisions[field_name].is_acceptable:
                    critical_ok = False
                    break
        
        # Calculate acceptance rate
        total = len(field_decisions)
        acceptable = sum(1 for d in field_decisions.values() if d.is_acceptable)
        acceptance_rate = acceptable / total if total > 0 else 0.0
        
        # Determine overall decision
        if errors:
            overall = Decision.REJECTED
        elif not critical_ok:
            overall = Decision.REVIEW_REQUIRED
        elif acceptance_rate >= 0.95:
            overall = Decision.VERIFIED
        elif acceptance_rate >= self.config.min_acceptance_rate:
            overall = Decision.LIKELY
        elif acceptance_rate >= 0.5:
            overall = Decision.REVIEW_REQUIRED
        else:
            overall = Decision.REJECTED
        
        return DocumentDecision(
            document_id=document_id,
            source_file=source_file,
            field_decisions=field_decisions,
            overall_decision=overall,
            extraction_complete=len(errors) == 0,
            warnings=warnings,
            errors=errors,
            processing_metadata=extraction_metadata or {},
        )
    
    def _calculate_adjusted_score(
        self,
        base_confidence: float,
        reasons: List[DecisionReason],
    ) -> float:
        """Calculate adjusted confidence score based on reasons."""
        adjustment = sum(r.weight for r in reasons)
        adjusted = base_confidence + adjustment
        return max(0.0, min(1.0, adjusted))
    
    def _score_to_decision(
        self,
        score: float,
        thresholds: Dict[str, float],
    ) -> Decision:
        """Convert adjusted score to decision."""
        if score >= thresholds['verified']:
            return Decision.VERIFIED
        elif score >= thresholds['likely']:
            return Decision.LIKELY
        elif score >= thresholds['review']:
            return Decision.REVIEW_REQUIRED
        else:
            return Decision.REJECTED
    
    def batch_decide(
        self,
        extractions: List[Dict[str, Any]],
    ) -> List[DocumentDecision]:
        """
        Make decisions for multiple documents.
        
        Args:
            extractions: List of extraction results
            
        Returns:
            List of DocumentDecision objects
        """
        decisions = []
        
        for i, extraction in enumerate(extractions):
            doc_id = extraction.get('document_id', f'doc_{i}')
            source = extraction.get('source_file', 'unknown')
            fields = extraction.get('fields', {})
            confidences = extraction.get('confidences', {})
            
            # Create field decisions
            field_decisions = {}
            for field_name, value in fields.items():
                confidence = confidences.get(field_name, 0.5)
                field_decision = self.decide_field(
                    field_name=field_name,
                    value=value,
                    confidence=confidence,
                )
                field_decisions[field_name] = field_decision
            
            # Create document decision
            doc_decision = self.decide_document(
                document_id=doc_id,
                source_file=source,
                field_decisions=field_decisions,
                warnings=extraction.get('warnings', []),
                errors=extraction.get('errors', []),
            )
            
            decisions.append(doc_decision)
        
        return decisions
