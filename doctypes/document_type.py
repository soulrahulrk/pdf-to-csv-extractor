"""
Document Type Definition

Defines the structure for configurable document types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Pattern, Callable, Union


class FieldType(Enum):
    """Types of fields that can be extracted."""
    
    TEXT = auto()           # Free text
    NUMBER = auto()         # Numeric value
    CURRENCY = auto()       # Monetary amount
    DATE = auto()           # Date value
    EMAIL = auto()          # Email address
    PHONE = auto()          # Phone number
    ADDRESS = auto()        # Postal address
    PERCENTAGE = auto()     # Percentage value
    IDENTIFIER = auto()     # ID number (invoice#, account#, etc.)
    TABLE = auto()          # Tabular data
    LIST = auto()           # List of values
    BOOLEAN = auto()        # Yes/No, True/False
    
    @property
    def is_numeric(self) -> bool:
        """Whether this field type is numeric."""
        return self in (FieldType.NUMBER, FieldType.CURRENCY, FieldType.PERCENTAGE)


@dataclass
class ValidationRule:
    """A validation rule for a field."""
    
    name: str
    check: Callable[[Any], bool]
    error_message: str
    severity: str = 'error'  # 'error', 'warning', 'info'


@dataclass
class FieldDefinition:
    """
    Definition of a field to extract from a document.
    """
    
    # Basic info
    name: str
    field_type: FieldType
    description: str = ''
    
    # Extraction hints
    labels: List[str] = field(default_factory=list)  # Label text to look for
    patterns: List[str] = field(default_factory=list)  # Regex patterns
    position_hint: Optional[str] = None  # 'top', 'bottom', 'left', 'right'
    
    # Validation
    required: bool = False
    min_confidence: float = 0.5
    validation_rules: List[ValidationRule] = field(default_factory=list)
    
    # Format
    format_pattern: Optional[str] = None  # Expected format regex
    normalize: bool = True  # Whether to normalize the value
    
    # Metadata
    aliases: List[str] = field(default_factory=list)  # Alternative names
    group: Optional[str] = None  # Field group (e.g., 'header', 'totals')
    depends_on: List[str] = field(default_factory=list)  # Dependencies
    
    # Computed
    _compiled_patterns: List[Pattern] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """Compile regex patterns."""
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]
    
    def matches_label(self, text: str) -> bool:
        """Check if text matches any label."""
        text_lower = text.lower().strip()
        return any(
            label.lower() in text_lower
            for label in self.labels
        )
    
    def matches_pattern(self, text: str) -> Optional[str]:
        """Check if text matches any pattern, return match."""
        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None
    
    def validate(self, value: Any) -> List[str]:
        """
        Validate a value against rules.
        
        Returns list of error messages.
        """
        errors = []
        
        for rule in self.validation_rules:
            try:
                if not rule.check(value):
                    errors.append(rule.error_message)
            except Exception as e:
                errors.append(f"Validation error: {e}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'field_type': self.field_type.name,
            'description': self.description,
            'labels': self.labels,
            'patterns': self.patterns,
            'required': self.required,
            'min_confidence': self.min_confidence,
            'group': self.group,
        }


@dataclass
class DocumentConfig:
    """
    Configuration for document processing.
    """
    
    # OCR settings
    enable_ocr: bool = True
    ocr_dpi: int = 300
    ocr_language: str = 'eng'
    
    # Layout settings
    use_layout_analysis: bool = True
    table_detection: bool = True
    
    # Validation settings
    strict_validation: bool = False
    auto_correct: bool = False
    
    # Performance
    max_pages: Optional[int] = None
    timeout_per_page: float = 30.0
    
    # Output
    include_confidence: bool = True
    include_bboxes: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enable_ocr': self.enable_ocr,
            'ocr_dpi': self.ocr_dpi,
            'ocr_language': self.ocr_language,
            'use_layout_analysis': self.use_layout_analysis,
            'table_detection': self.table_detection,
            'strict_validation': self.strict_validation,
            'max_pages': self.max_pages,
        }


@dataclass
class DocumentType:
    """
    Definition of a document type.
    
    Specifies what fields to extract, how to identify the document,
    and validation rules.
    """
    
    # Identity
    name: str
    display_name: str
    description: str = ''
    version: str = '1.0'
    
    # Fields
    fields: List[FieldDefinition] = field(default_factory=list)
    
    # Identification
    identification_patterns: List[str] = field(default_factory=list)
    identification_keywords: List[str] = field(default_factory=list)
    min_identification_score: float = 0.6
    
    # Table handling
    table_fields: List[str] = field(default_factory=list)  # Fields that are tables
    table_column_mappings: Dict[str, List[str]] = field(default_factory=dict)
    
    # Configuration
    config: DocumentConfig = field(default_factory=DocumentConfig)
    
    # Validation
    cross_field_rules: List[Callable[[Dict[str, Any]], List[str]]] = field(
        default_factory=list
    )
    
    # Metadata
    category: str = 'general'
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed properties."""
        self._field_map = {f.name: f for f in self.fields}
        self._required_fields = [f.name for f in self.fields if f.required]
        self._identification_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.identification_patterns
        ]
    
    @property
    def required_fields(self) -> List[str]:
        """Get list of required field names."""
        return self._required_fields
    
    @property
    def field_names(self) -> List[str]:
        """Get all field names."""
        return [f.name for f in self.fields]
    
    def get_field(self, name: str) -> Optional[FieldDefinition]:
        """Get field definition by name."""
        return self._field_map.get(name)
    
    def add_field(self, field_def: FieldDefinition) -> None:
        """Add a field definition."""
        self.fields.append(field_def)
        self._field_map[field_def.name] = field_def
        if field_def.required:
            self._required_fields.append(field_def.name)
    
    def get_fields_by_group(self, group: str) -> List[FieldDefinition]:
        """Get fields belonging to a group."""
        return [f for f in self.fields if f.group == group]
    
    def identify_document(self, text: str) -> float:
        """
        Score how likely a document is of this type.
        
        Args:
            text: Document text content
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0
        max_score = 0.0
        
        # Check patterns
        for pattern in self._identification_compiled:
            max_score += 1.0
            if pattern.search(text):
                score += 1.0
        
        # Check keywords
        text_lower = text.lower()
        for keyword in self.identification_keywords:
            max_score += 0.5
            if keyword.lower() in text_lower:
                score += 0.5
        
        if max_score == 0:
            return 0.0
        
        return score / max_score
    
    def is_document_type(self, text: str) -> bool:
        """Check if document matches this type."""
        return self.identify_document(text) >= self.min_identification_score
    
    def validate_extraction(
        self,
        extracted: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Validate extracted data.
        
        Args:
            extracted: Dictionary of extracted field values
            
        Returns:
            Dictionary of field -> error messages
        """
        errors: Dict[str, List[str]] = {}
        
        # Check required fields
        for field_name in self._required_fields:
            if field_name not in extracted or extracted[field_name] is None:
                errors[field_name] = [f"Required field '{field_name}' is missing"]
        
        # Validate individual fields
        for field_def in self.fields:
            if field_def.name in extracted:
                field_errors = field_def.validate(extracted[field_def.name])
                if field_errors:
                    errors[field_def.name] = field_errors
        
        # Cross-field validation
        for rule in self.cross_field_rules:
            try:
                rule_errors = rule(extracted)
                for error in rule_errors:
                    if '_general' not in errors:
                        errors['_general'] = []
                    errors['_general'].append(error)
            except Exception as e:
                if '_general' not in errors:
                    errors['_general'] = []
                errors['_general'].append(f"Cross-field validation error: {e}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'fields': [f.to_dict() for f in self.fields],
            'identification_patterns': self.identification_patterns,
            'identification_keywords': self.identification_keywords,
            'config': self.config.to_dict(),
            'category': self.category,
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentType':
        """Create from dictionary."""
        fields = []
        for f_data in data.get('fields', []):
            fields.append(FieldDefinition(
                name=f_data['name'],
                field_type=FieldType[f_data['field_type']],
                description=f_data.get('description', ''),
                labels=f_data.get('labels', []),
                patterns=f_data.get('patterns', []),
                required=f_data.get('required', False),
                min_confidence=f_data.get('min_confidence', 0.5),
                group=f_data.get('group'),
            ))
        
        config_data = data.get('config', {})
        config = DocumentConfig(
            enable_ocr=config_data.get('enable_ocr', True),
            ocr_dpi=config_data.get('ocr_dpi', 300),
            use_layout_analysis=config_data.get('use_layout_analysis', True),
            table_detection=config_data.get('table_detection', True),
        )
        
        return cls(
            name=data['name'],
            display_name=data.get('display_name', data['name']),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            fields=fields,
            identification_patterns=data.get('identification_patterns', []),
            identification_keywords=data.get('identification_keywords', []),
            config=config,
            category=data.get('category', 'general'),
            tags=data.get('tags', []),
        )


def create_field(
    name: str,
    field_type: Union[str, FieldType],
    labels: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    required: bool = False,
    **kwargs,
) -> FieldDefinition:
    """
    Convenience function to create a field definition.
    
    Args:
        name: Field name
        field_type: Field type (string or FieldType)
        labels: Label text to look for
        patterns: Regex patterns
        required: Whether field is required
        **kwargs: Additional field options
        
    Returns:
        FieldDefinition
    """
    if isinstance(field_type, str):
        field_type = FieldType[field_type.upper()]
    
    return FieldDefinition(
        name=name,
        field_type=field_type,
        labels=labels or [],
        patterns=patterns or [],
        required=required,
        **kwargs,
    )
