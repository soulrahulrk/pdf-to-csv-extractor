"""
Field Mapper Module

This module handles the mapping of extracted text to structured fields.
It uses the configuration from fields.yaml to find and extract specific
pieces of data from raw PDF text.

Architecture:
1. Load field definitions from YAML config
2. For each field, try patterns in priority order
3. Apply positional heuristics when patterns fail
4. Calculate confidence scores for each extraction

Why regex patterns aren't enough:
Real-world PDFs have inconsistent layouts. A field like "Invoice Date"
might appear as:
  - "Invoice Date: 2024-01-15"
  - "DATE          01/15/2024"
  - "Dated: January 15, 2024"
  - Or just "15-Jan-2024" near other invoice-related text

We combine multiple strategies to handle this variability.
"""

import re
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml
from loguru import logger


@dataclass
class FieldDefinition:
    """
    Definition of a field to extract from PDF text.
    Loaded from YAML configuration.
    """
    name: str                                   # Internal name (snake_case)
    display_name: str                           # Human-readable name
    field_type: str                             # string, number, currency, date, text_block
    required: bool = False                      # Whether missing field is an error
    patterns: list[str] = field(default_factory=list)  # Regex patterns to try
    keywords: list[str] = field(default_factory=list)  # Context keywords
    multiline: bool = False                     # Can span multiple lines
    validation: dict = field(default_factory=dict)     # Validation rules
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FieldDefinition':
        """Create FieldDefinition from config dict."""
        return cls(
            name=data.get('name', ''),
            display_name=data.get('display_name', data.get('name', '')),
            field_type=data.get('type', 'string'),
            required=data.get('required', False),
            patterns=data.get('patterns', []),
            keywords=data.get('keywords', []),
            multiline=data.get('multiline', False),
            validation=data.get('validation', {})
        )


@dataclass
class ExtractedField:
    """
    Result of extracting a single field from text.
    """
    name: str                           # Field name
    value: Any                          # Extracted value
    raw_value: str                      # Original matched text
    confidence: float                   # 0.0 to 1.0
    method: str                         # How it was found (pattern, keyword, position)
    pattern_used: Optional[str] = None  # Which pattern matched
    warnings: list[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if extraction was successful."""
        return self.value is not None and self.confidence > 0


class ConfigLoader:
    """
    Loads and parses field configuration from YAML files.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to fields.yaml config file
        """
        self.config_path = config_path
        self.config = {}
        self.fields: list[FieldDefinition] = []
        self.settings = {}
        self.noise_patterns = []
        self.line_items_config = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Path) -> dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Loaded configuration dict
        """
        logger.info(f"Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Parse settings
            self.settings = self.config.get('settings', {})
            
            # Parse field definitions
            self.fields = [
                FieldDefinition.from_dict(f) 
                for f in self.config.get('fields', [])
            ]
            
            # Parse noise patterns
            self.noise_patterns = self.config.get('noise_patterns', [])
            
            # Parse line items config
            self.line_items_config = self.config.get('line_items', {})
            
            logger.info(f"Loaded {len(self.fields)} field definitions")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def get_field(self, name: str) -> Optional[FieldDefinition]:
        """Get a field definition by name."""
        for field_def in self.fields:
            if field_def.name == name:
                return field_def
        return None
    
    def get_required_fields(self) -> list[FieldDefinition]:
        """Get list of required fields."""
        return [f for f in self.fields if f.required]
    
    def get_date_formats(self) -> list[str]:
        """Get configured date formats."""
        return self.settings.get('date_formats', [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"
        ])
    
    def get_currency_symbols(self) -> list[str]:
        """Get configured currency symbols."""
        return self.settings.get('currency_symbols', ['$', '€', '£'])
    
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold."""
        return self.settings.get('confidence_threshold', 0.6)


class FieldMapper:
    """
    Maps extracted text to structured fields using patterns and heuristics.
    
    Usage:
        mapper = FieldMapper(config_path=Path("config/fields.yaml"))
        results = mapper.extract_fields(text)
        for field in results:
            print(f"{field.name}: {field.value} (confidence: {field.confidence})")
    """
    
    def __init__(self, config_path: Optional[Path] = None, config: Optional[dict] = None):
        """
        Initialize field mapper.
        
        Args:
            config_path: Path to fields.yaml config
            config: Pre-loaded config dict (alternative to config_path)
        """
        self.config_loader = ConfigLoader()
        
        if config_path:
            self.config_loader.load(config_path)
        elif config:
            self.config_loader.config = config
            self.config_loader.fields = [
                FieldDefinition.from_dict(f) 
                for f in config.get('fields', [])
            ]
            self.config_loader.settings = config.get('settings', {})
            self.config_loader.noise_patterns = config.get('noise_patterns', [])
    
    def extract_fields(self, text: str) -> list[ExtractedField]:
        """
        Extract all configured fields from text.
        
        Args:
            text: Extracted PDF text to parse
            
        Returns:
            List of ExtractedField objects
        """
        if not text:
            logger.warning("Empty text provided for field extraction")
            return []
        
        results = []
        
        for field_def in self.config_loader.fields:
            result = self.extract_field(text, field_def)
            results.append(result)
            
            if result.is_valid:
                logger.debug(
                    f"Extracted {field_def.name}: '{result.value}' "
                    f"(confidence: {result.confidence:.2f})"
                )
            elif field_def.required:
                logger.warning(f"Required field '{field_def.name}' not found")
        
        return results
    
    def extract_field(self, text: str, field_def: FieldDefinition) -> ExtractedField:
        """
        Extract a single field from text.
        
        Strategy:
        1. Try each regex pattern in order
        2. If no pattern matches, use keyword proximity search
        3. Apply type-specific normalization
        4. Calculate confidence score
        """
        # Strategy 1: Pattern matching
        for pattern in field_def.patterns:
            result = self._try_pattern(text, pattern, field_def)
            if result.is_valid:
                return result
        
        # Strategy 2: Keyword proximity (fallback)
        if field_def.keywords:
            result = self._try_keyword_proximity(text, field_def)
            if result.is_valid:
                return result
        
        # Field not found
        return ExtractedField(
            name=field_def.name,
            value=None,
            raw_value="",
            confidence=0.0,
            method="not_found",
            warnings=[f"Could not extract '{field_def.display_name}'"]
        )
    
    def _try_pattern(
        self, 
        text: str, 
        pattern: str, 
        field_def: FieldDefinition
    ) -> ExtractedField:
        """
        Try to extract field using a regex pattern.
        """
        try:
            # Compile pattern with appropriate flags
            flags = re.IGNORECASE | re.MULTILINE
            if field_def.multiline:
                flags |= re.DOTALL
            
            regex = re.compile(pattern, flags)
            match = regex.search(text)
            
            if match:
                # Get the captured group (or full match if no groups)
                raw_value = match.group(1) if match.groups() else match.group(0)
                raw_value = raw_value.strip()
                
                # Clean up multiline values
                if field_def.multiline:
                    raw_value = self._clean_multiline(raw_value)
                
                # Calculate confidence based on match quality
                confidence = self._calculate_pattern_confidence(
                    raw_value, pattern, field_def
                )
                
                return ExtractedField(
                    name=field_def.name,
                    value=raw_value,
                    raw_value=raw_value,
                    confidence=confidence,
                    method="pattern",
                    pattern_used=pattern
                )
                
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        except Exception as e:
            logger.debug(f"Pattern matching failed: {e}")
        
        return ExtractedField(
            name=field_def.name,
            value=None,
            raw_value="",
            confidence=0.0,
            method="pattern_failed"
        )
    
    def _try_keyword_proximity(
        self, 
        text: str, 
        field_def: FieldDefinition
    ) -> ExtractedField:
        """
        Extract field by finding keywords and grabbing nearby content.
        
        This is a fallback when patterns don't match. It:
        1. Finds all occurrences of context keywords
        2. Extracts content that appears after/near keywords
        3. Applies type-specific extraction (date, currency, etc.)
        """
        best_match = None
        best_confidence = 0.0
        
        for keyword in field_def.keywords:
            # Find keyword in text (case insensitive)
            keyword_pattern = re.compile(
                rf'\b{re.escape(keyword)}\b[:\s]*(.+?)(?:\n|$)',
                re.IGNORECASE
            )
            
            for match in keyword_pattern.finditer(text):
                raw_value = match.group(1).strip()
                
                # For specific types, extract appropriately
                if field_def.field_type == 'date':
                    extracted = self._extract_date_near_position(text, match.start())
                    if extracted:
                        raw_value = extracted
                        
                elif field_def.field_type == 'currency':
                    extracted = self._extract_currency_near_position(text, match.start())
                    if extracted:
                        raw_value = extracted
                
                # Calculate confidence (lower for keyword method)
                confidence = self._calculate_keyword_confidence(
                    raw_value, keyword, field_def
                )
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = raw_value
        
        if best_match:
            return ExtractedField(
                name=field_def.name,
                value=best_match,
                raw_value=best_match,
                confidence=best_confidence,
                method="keyword"
            )
        
        return ExtractedField(
            name=field_def.name,
            value=None,
            raw_value="",
            confidence=0.0,
            method="keyword_failed"
        )
    
    def _extract_date_near_position(self, text: str, position: int) -> Optional[str]:
        """
        Extract a date-like string near a given position in the text.
        """
        # Look at text window around the position
        start = max(0, position - 20)
        end = min(len(text), position + 100)
        window = text[start:end]
        
        # Date patterns to look for
        date_patterns = [
            r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',
            r'\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}',
            r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, window)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_currency_near_position(self, text: str, position: int) -> Optional[str]:
        """
        Extract a currency value near a given position.
        """
        start = max(0, position - 20)
        end = min(len(text), position + 100)
        window = text[start:end]
        
        # Currency patterns
        currency_pattern = r'[$€£¥₹]?\s*[\d,]+\.?\d*'
        match = re.search(currency_pattern, window)
        
        if match:
            return match.group(0)
        
        return None
    
    def _clean_multiline(self, value: str) -> str:
        """Clean up multiline field values."""
        # Normalize line endings
        value = value.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines
        value = re.sub(r'\n{3,}', '\n\n', value)
        
        # Strip each line
        lines = [line.strip() for line in value.split('\n')]
        
        # Remove empty trailing/leading lines
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        
        return '\n'.join(lines)
    
    def _calculate_pattern_confidence(
        self, 
        value: str, 
        pattern: str, 
        field_def: FieldDefinition
    ) -> float:
        """
        Calculate confidence score for pattern-based extraction.
        
        Factors:
        - Value length (too short or too long is suspicious)
        - Presence of garbage characters
        - Match with expected format for field type
        """
        if not value:
            return 0.0
        
        confidence = 0.85  # Base confidence for pattern match
        
        # Length checks
        min_len = field_def.validation.get('min_length', 1)
        max_len = field_def.validation.get('max_length', 1000)
        
        if len(value) < min_len:
            confidence -= 0.3
        elif len(value) > max_len:
            confidence -= 0.2
        
        # Check for garbage characters
        garbage_ratio = sum(
            1 for c in value 
            if not c.isalnum() and c not in ' .,/-:\'\"@#$%&()'
        ) / max(len(value), 1)
        
        if garbage_ratio > 0.2:
            confidence -= 0.2
        
        # Type-specific validation
        if field_def.field_type == 'date':
            if not re.search(r'\d', value):
                confidence -= 0.3
                
        elif field_def.field_type == 'currency':
            if not re.search(r'\d', value):
                confidence -= 0.4
                
        elif field_def.field_type == 'number':
            cleaned = re.sub(r'[,\s$€£]', '', value)
            try:
                float(cleaned)
            except ValueError:
                confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_keyword_confidence(
        self, 
        value: str, 
        keyword: str, 
        field_def: FieldDefinition
    ) -> float:
        """
        Calculate confidence for keyword-based extraction.
        Lower base confidence since this is a fallback method.
        """
        if not value:
            return 0.0
        
        # Lower base confidence for keyword method
        confidence = 0.65
        
        # Apply similar checks as pattern confidence
        min_len = field_def.validation.get('min_length', 1)
        max_len = field_def.validation.get('max_length', 1000)
        
        if len(value) < min_len or len(value) > max_len:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def get_extraction_summary(
        self, 
        results: list[ExtractedField]
    ) -> dict[str, Any]:
        """
        Generate a summary of extraction results.
        
        Returns:
            Dict with extraction statistics and warnings
        """
        total = len(results)
        extracted = sum(1 for r in results if r.is_valid)
        required_missing = [
            r.name for r in results 
            if not r.is_valid and self.config_loader.get_field(r.name).required
        ]
        
        avg_confidence = (
            sum(r.confidence for r in results if r.is_valid) / max(extracted, 1)
        )
        
        return {
            'total_fields': total,
            'extracted_count': extracted,
            'extraction_rate': extracted / max(total, 1),
            'average_confidence': round(avg_confidence, 3),
            'required_missing': required_missing,
            'warnings': [
                w for r in results for w in r.warnings
            ],
            'fields': {
                r.name: {
                    'value': r.value,
                    'confidence': r.confidence,
                    'method': r.method
                } for r in results
            }
        }


def extract_fields_from_text(
    text: str, 
    config_path: Path
) -> tuple[dict[str, Any], list[str]]:
    """
    Convenience function to extract fields from text.
    
    Args:
        text: PDF text content
        config_path: Path to fields.yaml
        
    Returns:
        Tuple of (extracted_data_dict, list_of_warnings)
    """
    mapper = FieldMapper(config_path=config_path)
    results = mapper.extract_fields(text)
    
    data = {r.name: r.value for r in results if r.is_valid}
    warnings = [w for r in results for w in r.warnings]
    
    return data, warnings
