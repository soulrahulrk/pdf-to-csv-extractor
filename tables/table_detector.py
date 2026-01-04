"""
Table Detection

Identifies tables within PDF pages using structural analysis.
Handles various table formats including bordered, borderless, and hybrid tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Iterator

logger = logging.getLogger(__name__)


class TableType(Enum):
    """Types of tables based on structure."""
    
    BORDERED = auto()       # Tables with visible borders/lines
    BORDERLESS = auto()     # Tables using whitespace alignment
    HYBRID = auto()         # Mix of borders and whitespace
    UNKNOWN = auto()
    
    @property
    def needs_alignment_analysis(self) -> bool:
        """Whether this table type requires column alignment analysis."""
        return self in (TableType.BORDERLESS, TableType.HYBRID, TableType.UNKNOWN)


@dataclass
class TableBoundary:
    """
    Represents the detected boundary of a table on a page.
    """
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    table_type: TableType
    confidence: float
    row_count_estimate: int = 0
    column_count_estimate: int = 0
    has_header: bool = False
    header_row_count: int = 0
    is_continuation: bool = False
    continues_on_next: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def width(self) -> float:
        """Table width."""
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        """Table height."""
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        """Table area."""
        return self.width * self.height
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as tuple."""
        return (self.x0, self.y0, self.x1, self.y1)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside table boundary."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
    
    def overlaps(self, other: 'TableBoundary') -> bool:
        """Check if tables overlap."""
        return not (
            self.x1 < other.x0 or
            self.x0 > other.x1 or
            self.y1 < other.y0 or
            self.y0 > other.y1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'page_number': self.page_number,
            'bbox': self.bbox,
            'table_type': self.table_type.name,
            'confidence': round(self.confidence, 3),
            'row_count_estimate': self.row_count_estimate,
            'column_count_estimate': self.column_count_estimate,
            'has_header': self.has_header,
            'header_row_count': self.header_row_count,
            'is_continuation': self.is_continuation,
            'continues_on_next': self.continues_on_next,
        }


@dataclass
class DetectedLine:
    """A line detected in the document."""
    x0: float
    y0: float
    x1: float
    y1: float
    is_horizontal: bool
    is_vertical: bool
    length: float
    page_number: int


@dataclass
class ColumnAlignment:
    """Column alignment information for borderless tables."""
    column_index: int
    left_x: float
    right_x: float
    center_x: float
    alignment_type: str  # 'left', 'right', 'center', 'decimal'
    confidence: float


class TableDetector:
    """
    Detects tables within PDF pages.
    
    Uses multiple strategies:
    1. Line-based detection for bordered tables
    2. Whitespace/alignment analysis for borderless tables
    3. Structural heuristics for hybrid tables
    
    Usage:
        detector = TableDetector()
        
        # Detect from pdfplumber page
        tables = detector.detect_from_pdfplumber(page)
        
        # Detect from text blocks
        tables = detector.detect_from_blocks(blocks)
    """
    
    def __init__(
        self,
        min_rows: int = 2,
        min_columns: int = 2,
        min_confidence: float = 0.5,
        horizontal_tolerance: float = 5.0,
        vertical_tolerance: float = 5.0,
    ):
        """
        Initialize table detector.
        
        Args:
            min_rows: Minimum rows to consider as table
            min_columns: Minimum columns to consider as table
            min_confidence: Minimum confidence threshold
            horizontal_tolerance: Tolerance for horizontal alignment
            vertical_tolerance: Tolerance for vertical alignment
        """
        self.min_rows = min_rows
        self.min_columns = min_columns
        self.min_confidence = min_confidence
        self.horizontal_tolerance = horizontal_tolerance
        self.vertical_tolerance = vertical_tolerance
    
    def detect_from_pdfplumber(
        self,
        page: Any,
        page_number: int = 0,
    ) -> List[TableBoundary]:
        """
        Detect tables from a pdfplumber page.
        
        Args:
            page: pdfplumber page object
            page_number: Page number for reference
            
        Returns:
            List of detected table boundaries
        """
        tables: List[TableBoundary] = []
        
        # Try pdfplumber's built-in table detection first
        try:
            plumber_tables = page.find_tables()
            
            for i, table in enumerate(plumber_tables):
                bbox = table.bbox
                
                # Analyze table structure
                has_header, header_rows = self._detect_header(table)
                row_count = len(table.rows) if hasattr(table, 'rows') else 0
                col_count = len(table.cells[0]) if table.cells else 0
                
                boundary = TableBoundary(
                    page_number=page_number,
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3],
                    table_type=TableType.BORDERED,
                    confidence=0.85,
                    row_count_estimate=row_count,
                    column_count_estimate=col_count,
                    has_header=has_header,
                    header_row_count=header_rows,
                )
                
                tables.append(boundary)
                
        except Exception as e:
            logger.warning(f"pdfplumber table detection failed: {e}")
        
        # Also detect borderless tables
        try:
            borderless = self._detect_borderless_tables(page, page_number)
            
            # Only add if not overlapping with detected tables
            for bt in borderless:
                if not any(t.overlaps(bt) for t in tables):
                    tables.append(bt)
                    
        except Exception as e:
            logger.warning(f"Borderless table detection failed: {e}")
        
        return tables
    
    def detect_from_blocks(
        self,
        blocks: List[Any],
        page_number: int = 0,
        page_width: float = 612.0,
        page_height: float = 792.0,
    ) -> List[TableBoundary]:
        """
        Detect tables from text blocks.
        
        Args:
            blocks: List of TextBlock objects
            page_number: Page number for reference
            page_width: Page width
            page_height: Page height
            
        Returns:
            List of detected table boundaries
        """
        if not blocks:
            return []
        
        # Group blocks by vertical position
        row_groups = self._group_into_rows(blocks)
        
        # Find column alignments
        columns = self._detect_column_alignments(row_groups)
        
        if len(columns) < self.min_columns:
            return []
        
        if len(row_groups) < self.min_rows:
            return []
        
        # Calculate table boundary
        all_boxes = [b.bbox for b in blocks]
        x0 = min(box.x0 for box in all_boxes)
        y0 = min(box.y0 for box in all_boxes)
        x1 = max(box.x1 for box in all_boxes)
        y1 = max(box.y1 for box in all_boxes)
        
        # Estimate confidence based on alignment quality
        alignment_scores = [c.confidence for c in columns]
        confidence = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
        
        # Detect header
        has_header = self._detect_header_from_blocks(row_groups, columns)
        
        boundary = TableBoundary(
            page_number=page_number,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            table_type=TableType.BORDERLESS,
            confidence=confidence,
            row_count_estimate=len(row_groups),
            column_count_estimate=len(columns),
            has_header=has_header,
            header_row_count=1 if has_header else 0,
        )
        
        return [boundary]
    
    def _detect_borderless_tables(
        self,
        page: Any,
        page_number: int,
    ) -> List[TableBoundary]:
        """Detect tables without visible borders."""
        tables = []
        
        try:
            # Get all text with positioning
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
            )
            
            if len(words) < self.min_rows * self.min_columns:
                return []
            
            # Group words into rows by y-position
            rows = self._cluster_by_position(words, 'top', self.vertical_tolerance)
            
            if len(rows) < self.min_rows:
                return []
            
            # Find potential column boundaries
            column_positions = self._find_column_positions(rows)
            
            if len(column_positions) < self.min_columns:
                return []
            
            # Verify table structure
            structure_score = self._verify_table_structure(rows, column_positions)
            
            if structure_score >= self.min_confidence:
                # Calculate bounding box
                all_words = [w for row in rows for w in row]
                x0 = min(w['x0'] for w in all_words)
                y0 = min(w['top'] for w in all_words)
                x1 = max(w['x1'] for w in all_words)
                y1 = max(w['bottom'] for w in all_words)
                
                boundary = TableBoundary(
                    page_number=page_number,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    table_type=TableType.BORDERLESS,
                    confidence=structure_score,
                    row_count_estimate=len(rows),
                    column_count_estimate=len(column_positions),
                    has_header=True,  # Assume header for borderless
                    header_row_count=1,
                )
                tables.append(boundary)
        
        except Exception as e:
            logger.debug(f"Borderless detection error: {e}")
        
        return tables
    
    def _detect_header(self, table: Any) -> Tuple[bool, int]:
        """
        Detect if table has a header row.
        
        Args:
            table: pdfplumber table object
            
        Returns:
            (has_header, header_row_count)
        """
        try:
            if not table.cells or len(table.cells) < 2:
                return False, 0
            
            first_row = table.cells[0]
            second_row = table.cells[1] if len(table.cells) > 1 else None
            
            # Check if first row looks like headers
            header_indicators = 0
            
            # Headers often have different formatting
            # Check for bold text, different font size, etc.
            # This is a simplified heuristic
            
            for cell in first_row:
                if cell:
                    text = str(cell).strip()
                    # Headers are often short labels
                    if len(text) < 30 and text:
                        header_indicators += 1
                    # Headers rarely contain numbers
                    if not any(c.isdigit() for c in text):
                        header_indicators += 1
            
            # More than half the cells look like headers
            has_header = header_indicators >= len(first_row)
            
            return has_header, 1 if has_header else 0
            
        except Exception:
            return False, 0
    
    def _detect_header_from_blocks(
        self,
        row_groups: List[List[Any]],
        columns: List[ColumnAlignment],
    ) -> bool:
        """Detect header from text blocks."""
        if not row_groups:
            return False
        
        first_row = row_groups[0]
        
        # Headers typically:
        # - Don't contain many numbers
        # - Are shorter text
        # - Have different formatting
        
        numeric_count = 0
        total_count = len(first_row)
        
        for block in first_row:
            text = block.text.strip()
            if any(c.isdigit() for c in text):
                numeric_count += 1
        
        # If less than half contain numbers, likely a header
        return numeric_count < total_count / 2
    
    def _group_into_rows(
        self,
        blocks: List[Any],
    ) -> List[List[Any]]:
        """Group blocks into rows by y-position."""
        if not blocks:
            return []
        
        # Sort by y position
        sorted_blocks = sorted(blocks, key=lambda b: b.bbox.y0)
        
        rows = []
        current_row = [sorted_blocks[0]]
        current_y = sorted_blocks[0].bbox.y0
        
        for block in sorted_blocks[1:]:
            if abs(block.bbox.y0 - current_y) <= self.vertical_tolerance:
                current_row.append(block)
            else:
                # Sort row by x position
                current_row.sort(key=lambda b: b.bbox.x0)
                rows.append(current_row)
                current_row = [block]
                current_y = block.bbox.y0
        
        if current_row:
            current_row.sort(key=lambda b: b.bbox.x0)
            rows.append(current_row)
        
        return rows
    
    def _detect_column_alignments(
        self,
        row_groups: List[List[Any]],
    ) -> List[ColumnAlignment]:
        """Detect column alignments from row groups."""
        if not row_groups:
            return []
        
        # Collect all x positions
        all_positions = []
        for row in row_groups:
            for block in row:
                all_positions.append({
                    'left': block.bbox.x0,
                    'right': block.bbox.x1,
                    'center': (block.bbox.x0 + block.bbox.x1) / 2,
                })
        
        if not all_positions:
            return []
        
        # Cluster positions to find columns
        left_clusters = self._cluster_values(
            [p['left'] for p in all_positions],
            self.horizontal_tolerance,
        )
        
        columns = []
        for i, (center, positions) in enumerate(left_clusters):
            confidence = len(positions) / len(row_groups)
            
            columns.append(ColumnAlignment(
                column_index=i,
                left_x=center,
                right_x=center + 100,  # Estimate
                center_x=center + 50,
                alignment_type='left',
                confidence=min(1.0, confidence),
            ))
        
        return columns
    
    def _cluster_by_position(
        self,
        items: List[Dict],
        position_key: str,
        tolerance: float,
    ) -> List[List[Dict]]:
        """Cluster items by position."""
        if not items:
            return []
        
        sorted_items = sorted(items, key=lambda x: x[position_key])
        
        clusters = []
        current_cluster = [sorted_items[0]]
        current_pos = sorted_items[0][position_key]
        
        for item in sorted_items[1:]:
            if abs(item[position_key] - current_pos) <= tolerance:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
                current_pos = item[position_key]
        
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    def _cluster_values(
        self,
        values: List[float],
        tolerance: float,
    ) -> List[Tuple[float, List[float]]]:
        """Cluster numeric values."""
        if not values:
            return []
        
        sorted_values = sorted(values)
        
        clusters = []
        current_cluster = [sorted_values[0]]
        
        for value in sorted_values[1:]:
            if value - current_cluster[-1] <= tolerance:
                current_cluster.append(value)
            else:
                center = sum(current_cluster) / len(current_cluster)
                clusters.append((center, current_cluster))
                current_cluster = [value]
        
        if current_cluster:
            center = sum(current_cluster) / len(current_cluster)
            clusters.append((center, current_cluster))
        
        return clusters
    
    def _find_column_positions(
        self,
        rows: List[List[Dict]],
    ) -> List[float]:
        """Find column positions from rows of words."""
        all_x_positions = []
        
        for row in rows:
            for word in row:
                all_x_positions.append(word['x0'])
        
        if not all_x_positions:
            return []
        
        clusters = self._cluster_values(all_x_positions, self.horizontal_tolerance)
        
        # Filter clusters that appear in most rows
        min_occurrences = len(rows) * 0.5
        column_positions = [
            center for center, values in clusters
            if len(values) >= min_occurrences
        ]
        
        return sorted(column_positions)
    
    def _verify_table_structure(
        self,
        rows: List[List[Dict]],
        column_positions: List[float],
    ) -> float:
        """
        Verify that rows align with column positions.
        
        Returns a confidence score.
        """
        if not rows or not column_positions:
            return 0.0
        
        alignment_scores = []
        
        for row in rows:
            row_score = 0
            for word in row:
                x = word['x0']
                # Find closest column
                min_dist = min(abs(x - col) for col in column_positions)
                if min_dist <= self.horizontal_tolerance:
                    row_score += 1
            
            if row:
                alignment_scores.append(row_score / len(row))
        
        if alignment_scores:
            return sum(alignment_scores) / len(alignment_scores)
        
        return 0.0
    
    def detect_table_regions(
        self,
        page: Any,
        page_number: int = 0,
    ) -> List[TableBoundary]:
        """
        High-level method to detect all table regions.
        
        Combines multiple detection strategies.
        """
        all_tables = []
        
        # Try pdfplumber first
        plumber_tables = self.detect_from_pdfplumber(page, page_number)
        all_tables.extend(plumber_tables)
        
        # Merge overlapping detections
        merged = self._merge_overlapping(all_tables)
        
        return merged
    
    def _merge_overlapping(
        self,
        tables: List[TableBoundary],
    ) -> List[TableBoundary]:
        """Merge overlapping table boundaries."""
        if not tables:
            return []
        
        merged = []
        used = set()
        
        for i, t1 in enumerate(tables):
            if i in used:
                continue
            
            # Find all overlapping tables
            overlapping = [t1]
            
            for j, t2 in enumerate(tables[i+1:], i+1):
                if j not in used and t1.overlaps(t2):
                    overlapping.append(t2)
                    used.add(j)
            
            # Merge into single boundary
            if len(overlapping) > 1:
                x0 = min(t.x0 for t in overlapping)
                y0 = min(t.y0 for t in overlapping)
                x1 = max(t.x1 for t in overlapping)
                y1 = max(t.y1 for t in overlapping)
                confidence = max(t.confidence for t in overlapping)
                
                merged_table = TableBoundary(
                    page_number=t1.page_number,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    table_type=t1.table_type,
                    confidence=confidence,
                    row_count_estimate=max(t.row_count_estimate for t in overlapping),
                    column_count_estimate=max(t.column_count_estimate for t in overlapping),
                    has_header=any(t.has_header for t in overlapping),
                    header_row_count=max(t.header_row_count for t in overlapping),
                )
                merged.append(merged_table)
            else:
                merged.append(t1)
        
        return merged
