"""
Document Extractors

Extraction engine that uses document type definitions to extract structured data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import re
import logging

from .document_type import DocumentType, FieldDefinition, FieldType

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """How a field value was extracted."""
    LABEL_MATCH = "label_match"
    PATTERN_MATCH = "pattern_match"
    POSITION_BASED = "position_based"
    SPATIAL_RELATION = "spatial_relation"
    TABLE_CELL = "table_cell"
    OCR_REGION = "ocr_region"
    FALLBACK = "fallback"


@dataclass
class ExtractedValue:
    """A single extracted value with provenance."""
    
    value: str
    raw_value: str
    field_name: str
    confidence: float
    method: ExtractionMethod
    
    # Location info
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    
    # Provenance
    matched_label: Optional[str] = None
    matched_pattern: Optional[str] = None
    source_block_id: Optional[str] = None
    
    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'raw_value': self.raw_value,
            'field_name': self.field_name,
            'confidence': self.confidence,
            'method': self.method.value,
            'page': self.page,
            'bbox': self.bbox,
            'matched_label': self.matched_label,
            'matched_pattern': self.matched_pattern,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""
    
    document_type: str
    fields: Dict[str, ExtractedValue]
    tables: Dict[str, List[Dict[str, Any]]]
    
    # Metadata
    page_count: int = 0
    extraction_time_ms: float = 0
    
    # Quality metrics
    field_count: int = 0
    extracted_count: int = 0
    required_missing: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    @property
    def extraction_rate(self) -> float:
        """Percentage of fields successfully extracted."""
        if self.field_count == 0:
            return 0.0
        return self.extracted_count / self.field_count
    
    @property
    def is_complete(self) -> bool:
        """Whether all required fields were extracted."""
        return len(self.required_missing) == 0
    
    def get_value(self, field_name: str) -> Optional[str]:
        """Get extracted value by field name."""
        if field_name in self.fields:
            return self.fields[field_name].value
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'document_type': self.document_type,
            'fields': {k: v.to_dict() for k, v in self.fields.items()},
            'tables': self.tables,
            'page_count': self.page_count,
            'extraction_time_ms': self.extraction_time_ms,
            'field_count': self.field_count,
            'extracted_count': self.extracted_count,
            'extraction_rate': self.extraction_rate,
            'required_missing': self.required_missing,
            'validation_errors': self.validation_errors,
            'is_complete': self.is_complete,
        }
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary with just field values."""
        result = {'document_type': self.document_type}
        for name, extracted in self.fields.items():
            result[name] = extracted.value
        return result


@dataclass
class TextBlock:
    """A text block from the document."""
    id: str
    text: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    confidence: float = 1.0
    block_type: str = "text"


class DocumentExtractor:
    """
    Extracts structured data from documents using type definitions.
    
    Uses multiple extraction strategies:
    1. Label-based: Look for labels near values
    2. Pattern-based: Match regex patterns
    3. Position-based: Use spatial hints
    4. Relation-based: Use spatial relationships between blocks
    """
    
    def __init__(
        self,
        document_type: DocumentType,
        fuzzy_threshold: float = 0.8,
        use_ocr: bool = True,
    ):
        self.document_type = document_type
        self.fuzzy_threshold = fuzzy_threshold
        self.use_ocr = use_ocr
        
        # Build label lookup for fast matching
        self._label_to_field = self._build_label_index()
    
    def _build_label_index(self) -> Dict[str, str]:
        """Build index of normalized labels to field names."""
        index = {}
        for field_def in self.document_type.fields:
            for label in field_def.labels:
                normalized = self._normalize_label(label)
                index[normalized] = field_def.name
        return index
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label for matching."""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = label.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _fuzzy_match_label(self, text: str) -> Optional[Tuple[str, float]]:
        """Fuzzy match text against known labels."""
        normalized = self._normalize_label(text)
        
        # Exact match first
        if normalized in self._label_to_field:
            return self._label_to_field[normalized], 1.0
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        
        for label, field_name in self._label_to_field.items():
            score = self._string_similarity(normalized, label)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = field_name
        
        if best_match:
            return best_match, best_score
        return None
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein ratio."""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Simple containment check
        if s1 in s2 or s2 in s1:
            return 0.9
        
        # Levenshtein distance
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        # Use two-row algorithm for space efficiency
        prev_row = list(range(len1 + 1))
        
        for i, c2 in enumerate(s2):
            curr_row = [i + 1]
            for j, c1 in enumerate(s1):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        distance = prev_row[-1]
        max_len = max(len1, len2)
        return 1.0 - (distance / max_len)
    
    def extract(
        self,
        text_blocks: List[TextBlock],
        tables: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from document content.
        
        Args:
            text_blocks: List of text blocks with positions
            tables: Optional pre-extracted tables
        
        Returns:
            ExtractionResult with all extracted fields
        """
        import time
        start_time = time.time()
        
        fields: Dict[str, ExtractedValue] = {}
        extracted_tables: Dict[str, List[Dict[str, Any]]] = {}
        
        # Group blocks by page
        blocks_by_page = self._group_blocks_by_page(text_blocks)
        
        # Strategy 1: Label-based extraction
        self._extract_by_labels(text_blocks, fields)
        
        # Strategy 2: Pattern-based extraction
        self._extract_by_patterns(text_blocks, fields)
        
        # Strategy 3: Position-based extraction
        self._extract_by_position(blocks_by_page, fields)
        
        # Strategy 4: Extract tables
        if tables:
            extracted_tables = self._process_tables(tables)
        
        # Validate extracted values
        validation_errors = self._validate_extraction(fields)
        
        # Cross-field validation
        field_values = {k: v.value for k, v in fields.items()}
        for rule in self.document_type.cross_field_rules:
            try:
                errors = rule(field_values)
                validation_errors.extend(errors)
            except Exception as e:
                logger.warning(f"Cross-field rule error: {e}")
        
        # Check required fields
        required_missing = []
        for field_def in self.document_type.fields:
            if field_def.required and field_def.name not in fields:
                required_missing.append(field_def.name)
        
        # Calculate page count
        page_count = max(b.page for b in text_blocks) + 1 if text_blocks else 0
        
        # Calculate extraction time
        extraction_time_ms = (time.time() - start_time) * 1000
        
        return ExtractionResult(
            document_type=self.document_type.name,
            fields=fields,
            tables=extracted_tables,
            page_count=page_count,
            extraction_time_ms=extraction_time_ms,
            field_count=len(self.document_type.fields),
            extracted_count=len(fields),
            required_missing=required_missing,
            validation_errors=validation_errors,
        )
    
    def _group_blocks_by_page(
        self,
        blocks: List[TextBlock],
    ) -> Dict[int, List[TextBlock]]:
        """Group blocks by page number."""
        by_page: Dict[int, List[TextBlock]] = {}
        for block in blocks:
            if block.page not in by_page:
                by_page[block.page] = []
            by_page[block.page].append(block)
        return by_page
    
    def _extract_by_labels(
        self,
        blocks: List[TextBlock],
        fields: Dict[str, ExtractedValue],
    ) -> None:
        """Extract values by matching labels in adjacent blocks."""
        
        # Sort blocks by position (top to bottom, left to right)
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (b.page, b.bbox[1], b.bbox[0])
        )
        
        for i, block in enumerate(sorted_blocks):
            # Check if this block contains a label
            match = self._fuzzy_match_label(block.text)
            if not match:
                continue
            
            field_name, label_confidence = match
            
            # Skip if already extracted with higher confidence
            if field_name in fields and fields[field_name].confidence >= label_confidence:
                continue
            
            # Look for value in the same block (label: value format)
            value = self._extract_value_from_label_block(block, field_name)
            if value:
                fields[field_name] = value
                continue
            
            # Look for value in adjacent blocks
            value = self._find_adjacent_value(
                sorted_blocks, i, field_name, label_confidence
            )
            if value:
                fields[field_name] = value
    
    def _extract_value_from_label_block(
        self,
        block: TextBlock,
        field_name: str,
    ) -> Optional[ExtractedValue]:
        """Try to extract value from a label block (label: value format)."""
        
        # Common separators
        separators = [':', '-', '|', '\t']
        
        for sep in separators:
            if sep in block.text:
                parts = block.text.split(sep, 1)
                if len(parts) == 2:
                    potential_label = parts[0].strip()
                    potential_value = parts[1].strip()
                    
                    # Verify the label part matches
                    match = self._fuzzy_match_label(potential_label)
                    if match and match[0] == field_name and potential_value:
                        return ExtractedValue(
                            value=potential_value,
                            raw_value=potential_value,
                            field_name=field_name,
                            confidence=match[1] * 0.95,
                            method=ExtractionMethod.LABEL_MATCH,
                            page=block.page,
                            bbox=block.bbox,
                            matched_label=potential_label,
                            source_block_id=block.id,
                        )
        
        return None
    
    def _find_adjacent_value(
        self,
        blocks: List[TextBlock],
        label_idx: int,
        field_name: str,
        label_confidence: float,
    ) -> Optional[ExtractedValue]:
        """Find value in blocks adjacent to a label block."""
        
        label_block = blocks[label_idx]
        field_def = self.document_type.get_field(field_name)
        
        # Search nearby blocks (right and below)
        search_radius = 50  # pixels
        
        candidates: List[Tuple[TextBlock, float]] = []
        
        for i, block in enumerate(blocks):
            if i == label_idx:
                continue
            if block.page != label_block.page:
                continue
            
            # Skip blocks that look like labels
            if self._fuzzy_match_label(block.text):
                continue
            
            # Calculate spatial distance
            distance = self._calculate_distance(label_block, block)
            if distance > search_radius * 3:
                continue
            
            # Score based on position relationship
            score = self._score_value_position(label_block, block)
            if score > 0:
                candidates.append((block, score))
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_block, position_score = candidates[0]
        
        # Validate value format if patterns exist
        value_text = best_block.text.strip()
        if field_def and field_def.patterns:
            if not any(re.search(p, value_text, re.IGNORECASE) for p in field_def.patterns):
                # Pattern mismatch, reduce confidence
                position_score *= 0.7
        
        return ExtractedValue(
            value=value_text,
            raw_value=value_text,
            field_name=field_name,
            confidence=label_confidence * position_score,
            method=ExtractionMethod.LABEL_MATCH,
            page=best_block.page,
            bbox=best_block.bbox,
            matched_label=label_block.text,
            source_block_id=best_block.id,
        )
    
    def _calculate_distance(self, block1: TextBlock, block2: TextBlock) -> float:
        """Calculate distance between two blocks."""
        # Use center points
        c1_x = (block1.bbox[0] + block1.bbox[2]) / 2
        c1_y = (block1.bbox[1] + block1.bbox[3]) / 2
        c2_x = (block2.bbox[0] + block2.bbox[2]) / 2
        c2_y = (block2.bbox[1] + block2.bbox[3]) / 2
        
        return ((c2_x - c1_x) ** 2 + (c2_y - c1_y) ** 2) ** 0.5
    
    def _score_value_position(
        self,
        label_block: TextBlock,
        value_block: TextBlock,
    ) -> float:
        """Score how likely a value block relates to a label block."""
        
        label_bbox = label_block.bbox
        value_bbox = value_block.bbox
        
        # Values to the right are preferred
        if value_bbox[0] > label_bbox[2]:  # Value starts after label ends
            # Check vertical alignment
            label_mid_y = (label_bbox[1] + label_bbox[3]) / 2
            value_mid_y = (value_bbox[1] + value_bbox[3]) / 2
            
            vertical_diff = abs(label_mid_y - value_mid_y)
            if vertical_diff < 10:  # Well aligned
                return 0.95
            elif vertical_diff < 20:
                return 0.85
        
        # Values below are also acceptable
        if value_bbox[1] > label_bbox[3]:  # Value starts below label
            # Check horizontal alignment
            label_left = label_bbox[0]
            value_left = value_bbox[0]
            
            horizontal_diff = abs(value_left - label_left)
            if horizontal_diff < 20:  # Well aligned
                return 0.80
            elif horizontal_diff < 50:
                return 0.70
        
        # Other positions get lower scores
        return 0.3
    
    def _extract_by_patterns(
        self,
        blocks: List[TextBlock],
        fields: Dict[str, ExtractedValue],
    ) -> None:
        """Extract values using regex patterns."""
        
        for field_def in self.document_type.fields:
            # Skip if already extracted with good confidence
            if field_def.name in fields and fields[field_def.name].confidence > 0.8:
                continue
            
            if not field_def.patterns:
                continue
            
            for block in blocks:
                for pattern in field_def.patterns:
                    match = re.search(pattern, block.text, re.IGNORECASE)
                    if match:
                        value = match.group(0)
                        
                        # Calculate confidence based on pattern specificity
                        confidence = 0.75
                        if len(value) > 5:
                            confidence = 0.80
                        
                        # Only update if better than existing
                        if field_def.name not in fields or \
                           fields[field_def.name].confidence < confidence:
                            fields[field_def.name] = ExtractedValue(
                                value=value,
                                raw_value=value,
                                field_name=field_def.name,
                                confidence=confidence,
                                method=ExtractionMethod.PATTERN_MATCH,
                                page=block.page,
                                bbox=block.bbox,
                                matched_pattern=pattern,
                                source_block_id=block.id,
                            )
                        break  # Found match, move to next block
    
    def _extract_by_position(
        self,
        blocks_by_page: Dict[int, List[TextBlock]],
        fields: Dict[str, ExtractedValue],
    ) -> None:
        """Extract values based on position hints."""
        
        for field_def in self.document_type.fields:
            if field_def.name in fields:
                continue
            
            if not field_def.position_hint:
                continue
            
            # Get first page blocks for position-based extraction
            page_blocks = blocks_by_page.get(0, [])
            if not page_blocks:
                continue
            
            # Find blocks in the hinted position
            candidates = self._find_blocks_by_position(
                page_blocks, field_def.position_hint
            )
            
            for block in candidates:
                # Skip labels
                if self._fuzzy_match_label(block.text):
                    continue
                
                # Validate against patterns if available
                if field_def.patterns:
                    if not any(re.search(p, block.text, re.IGNORECASE) 
                              for p in field_def.patterns):
                        continue
                
                fields[field_def.name] = ExtractedValue(
                    value=block.text,
                    raw_value=block.text,
                    field_name=field_def.name,
                    confidence=0.6,  # Lower confidence for position-based
                    method=ExtractionMethod.POSITION_BASED,
                    page=block.page,
                    bbox=block.bbox,
                    source_block_id=block.id,
                )
                break
    
    def _find_blocks_by_position(
        self,
        blocks: List[TextBlock],
        position_hint: str,
    ) -> List[TextBlock]:
        """Find blocks in the specified position."""
        
        if not blocks:
            return []
        
        # Get page bounds
        all_x0 = [b.bbox[0] for b in blocks]
        all_y0 = [b.bbox[1] for b in blocks]
        all_x1 = [b.bbox[2] for b in blocks]
        all_y1 = [b.bbox[3] for b in blocks]
        
        page_left = min(all_x0)
        page_top = min(all_y0)
        page_right = max(all_x1)
        page_bottom = max(all_y1)
        
        page_width = page_right - page_left
        page_height = page_bottom - page_top
        
        # Define position regions
        regions = {
            'top': (page_top, page_top + page_height * 0.2),
            'bottom': (page_bottom - page_height * 0.2, page_bottom),
            'left': (page_left, page_left + page_width * 0.4),
            'right': (page_right - page_width * 0.4, page_right),
            'center': (page_left + page_width * 0.3, page_right - page_width * 0.3),
        }
        
        # Parse position hint
        parts = position_hint.lower().replace('-', ' ').split()
        
        result = blocks.copy()
        
        for part in parts:
            if part in ['top', 'bottom']:
                y_min, y_max = regions[part]
                result = [b for b in result 
                         if b.bbox[1] >= y_min and b.bbox[1] <= y_max]
            elif part in ['left', 'right', 'center']:
                x_min, x_max = regions[part]
                result = [b for b in result 
                         if b.bbox[0] >= x_min and b.bbox[2] <= x_max]
        
        # Sort by position
        result.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        
        return result
    
    def _process_tables(
        self,
        tables: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process and map tables to field definitions."""
        
        result: Dict[str, List[Dict[str, Any]]] = {}
        
        for table_field in self.document_type.table_fields:
            expected_columns = self.document_type.table_column_mappings.get(
                table_field, []
            )
            
            for table in tables:
                if self._table_matches_field(table, expected_columns):
                    if table_field not in result:
                        result[table_field] = []
                    result[table_field].append(table)
        
        return result
    
    def _table_matches_field(
        self,
        table: Dict[str, Any],
        expected_columns: List[str],
    ) -> bool:
        """Check if a table matches expected columns."""
        
        table_columns = [c.lower() for c in table.get('headers', [])]
        
        if not table_columns:
            return False
        
        # Check if any expected column is present
        for expected in expected_columns:
            expected_lower = expected.lower()
            for actual in table_columns:
                if expected_lower in actual or actual in expected_lower:
                    return True
        
        return False
    
    def _validate_extraction(
        self,
        fields: Dict[str, ExtractedValue],
    ) -> List[str]:
        """Validate extracted values against field definitions."""
        
        errors = []
        
        for field_name, extracted in fields.items():
            field_def = self.document_type.get_field(field_name)
            if not field_def:
                continue
            
            # Run custom validation rules
            for rule in field_def.validation_rules:
                try:
                    if not rule.validate(extracted.value):
                        extracted.is_valid = False
                        error_msg = f"{field_name}: {rule.error_message}"
                        extracted.validation_errors.append(error_msg)
                        errors.append(error_msg)
                except Exception as e:
                    logger.warning(f"Validation error for {field_name}: {e}")
        
        return errors


class BatchExtractor:
    """Extract data from multiple documents."""
    
    def __init__(self, extractor: DocumentExtractor):
        self.extractor = extractor
    
    def extract_batch(
        self,
        documents: List[List[TextBlock]],
    ) -> List[ExtractionResult]:
        """Extract from multiple documents."""
        return [
            self.extractor.extract(doc)
            for doc in documents
        ]
    
    def extract_to_csv(
        self,
        documents: List[List[TextBlock]],
        output_path: str,
    ) -> None:
        """Extract from documents and save to CSV."""
        import csv
        
        results = self.extract_batch(documents)
        
        if not results:
            return
        
        # Get all field names
        all_fields = set()
        for result in results:
            all_fields.update(result.fields.keys())
        
        fieldnames = ['document_type'] + sorted(all_fields)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = result.to_flat_dict()
                writer.writerow(row)
