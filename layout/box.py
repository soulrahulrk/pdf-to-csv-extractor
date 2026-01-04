"""
Bounding Box and Text Block Primitives

This module provides geometric primitives for representing text regions
in documents. These are the foundation for spatial reasoning.

Design Decisions:
- Coordinates are in PDF units (points, 1/72 inch) for consistency
- Origin is bottom-left (PDF convention) but we support top-left conversion
- Immutable design for thread safety and caching
- Rich comparison operators for sorting and spatial queries
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Iterator, Dict, Any
from functools import cached_property


class BlockType(Enum):
    """Classification of text block types."""
    TEXT = auto()           # Regular paragraph text
    HEADING = auto()        # Section headings
    LABEL = auto()          # Field labels (e.g., "Invoice Number:")
    VALUE = auto()          # Field values
    TABLE_CELL = auto()     # Cell within a table
    TABLE_HEADER = auto()   # Table column header
    LIST_ITEM = auto()      # Bulleted or numbered list item
    FOOTER = auto()         # Page footer
    HEADER = auto()         # Page header
    LOGO = auto()           # Logo region (usually image)
    SIGNATURE = auto()      # Signature region
    ANNOTATION = auto()     # Handwritten annotations
    NOISE = auto()          # OCR artifacts, watermarks
    UNKNOWN = auto()        # Unclassified


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable bounding box with PDF coordinate system.
    
    Coordinates:
    - x0, y0: Bottom-left corner (PDF convention)
    - x1, y1: Top-right corner
    
    All coordinates are in PDF points (1/72 inch).
    
    Thread Safety: Immutable, safe for concurrent access.
    
    Example:
        box = BoundingBox(x0=72, y0=700, x1=200, y1=720)
        print(f"Width: {box.width}, Height: {box.height}")
        
        # Check containment
        if point_box.is_inside(container_box):
            ...
            
        # Expand with margin
        expanded = box.expand(margin=5)
    """
    x0: float
    y0: float
    x1: float
    y1: float
    
    def __post_init__(self):
        """Validate coordinates on creation."""
        # Normalize coordinates if inverted
        if self.x0 > self.x1:
            object.__setattr__(self, 'x0', self.x1)
            object.__setattr__(self, 'x1', self.x0)
        if self.y0 > self.y1:
            object.__setattr__(self, 'y0', self.y1)
            object.__setattr__(self, 'y1', self.y0)
    
    @cached_property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x1 - self.x0
    
    @cached_property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y1 - self.y0
    
    @cached_property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    @cached_property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy)."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @cached_property
    def center_x(self) -> float:
        """X coordinate of center."""
        return (self.x0 + self.x1) / 2
    
    @cached_property
    def center_y(self) -> float:
        """Y coordinate of center."""
        return (self.y0 + self.y1) / 2
    
    @property
    def left(self) -> float:
        """Left edge (x0)."""
        return self.x0
    
    @property
    def right(self) -> float:
        """Right edge (x1)."""
        return self.x1
    
    @property
    def bottom(self) -> float:
        """Bottom edge (y0)."""
        return self.y0
    
    @property
    def top(self) -> float:
        """Top edge (y1)."""
        return self.y1
    
    @cached_property
    def diagonal(self) -> float:
        """Diagonal length."""
        return math.sqrt(self.width ** 2 + self.height ** 2)
    
    @cached_property
    def aspect_ratio(self) -> float:
        """Width / Height ratio. Returns inf for zero-height boxes."""
        if self.height == 0:
            return float('inf')
        return self.width / self.height
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (x0, y0, x1, y1) tuple."""
        return (self.x0, self.y0, self.x1, self.y1)
    
    def to_dict(self) -> Dict[str, float]:
        """Return as dictionary for JSON serialization."""
        return {
            'x0': round(self.x0, 2),
            'y0': round(self.y0, 2),
            'x1': round(self.x1, 2),
            'y1': round(self.y1, 2),
            'width': round(self.width, 2),
            'height': round(self.height, 2),
        }
    
    def to_top_left_origin(self, page_height: float) -> 'BoundingBox':
        """
        Convert from PDF (bottom-left origin) to image (top-left origin).
        
        Args:
            page_height: Height of the page for Y-axis inversion
            
        Returns:
            New BoundingBox with top-left origin coordinates
        """
        return BoundingBox(
            x0=self.x0,
            y0=page_height - self.y1,
            x1=self.x1,
            y1=page_height - self.y0,
        )
    
    def scale(self, factor: float) -> 'BoundingBox':
        """
        Scale the bounding box by a factor.
        
        Args:
            factor: Scaling factor (e.g., 2.0 for double size)
            
        Returns:
            New scaled BoundingBox
        """
        return BoundingBox(
            x0=self.x0 * factor,
            y0=self.y0 * factor,
            x1=self.x1 * factor,
            y1=self.y1 * factor,
        )
    
    def expand(self, margin: float) -> 'BoundingBox':
        """
        Expand box by margin on all sides.
        
        Args:
            margin: Pixels to expand (can be negative to shrink)
            
        Returns:
            New expanded BoundingBox
        """
        return BoundingBox(
            x0=self.x0 - margin,
            y0=self.y0 - margin,
            x1=self.x1 + margin,
            y1=self.y1 + margin,
        )
    
    def expand_asymmetric(
        self,
        left: float = 0,
        right: float = 0,
        top: float = 0,
        bottom: float = 0
    ) -> 'BoundingBox':
        """Expand with different margins per side."""
        return BoundingBox(
            x0=self.x0 - left,
            y0=self.y0 - bottom,
            x1=self.x1 + right,
            y1=self.y1 + top,
        )
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this box intersects with another."""
        return not (
            self.x1 < other.x0 or
            self.x0 > other.x1 or
            self.y1 < other.y0 or
            self.y0 > other.y1
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this box."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
    
    def contains_box(self, other: 'BoundingBox') -> bool:
        """Check if this box fully contains another box."""
        return (
            self.x0 <= other.x0 and
            self.y0 <= other.y0 and
            self.x1 >= other.x1 and
            self.y1 >= other.y1
        )
    
    def is_inside(self, other: 'BoundingBox') -> bool:
        """Check if this box is fully inside another box."""
        return other.contains_box(self)
    
    def intersection(self, other: 'BoundingBox') -> Optional['BoundingBox']:
        """
        Get the intersection of two boxes.
        
        Returns:
            Intersection BoundingBox or None if no intersection
        """
        if not self.intersects(other):
            return None
        
        return BoundingBox(
            x0=max(self.x0, other.x0),
            y0=max(self.y0, other.y0),
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
        )
    
    def union(self, other: 'BoundingBox') -> 'BoundingBox':
        """Get the smallest box containing both boxes."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """
        Minimum distance between edges of two boxes.
        Returns 0 if boxes intersect.
        """
        if self.intersects(other):
            return 0.0
        
        # Calculate gap in each direction
        dx = max(0, max(self.x0 - other.x1, other.x0 - self.x1))
        dy = max(0, max(self.y0 - other.y1, other.y0 - self.y1))
        
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def horizontal_distance_to(self, other: 'BoundingBox') -> float:
        """Horizontal gap between boxes (0 if overlapping horizontally)."""
        if self.x1 < other.x0:
            return other.x0 - self.x1
        elif other.x1 < self.x0:
            return self.x0 - other.x1
        return 0.0
    
    def vertical_distance_to(self, other: 'BoundingBox') -> float:
        """Vertical gap between boxes (0 if overlapping vertically)."""
        if self.y1 < other.y0:
            return other.y0 - self.y1
        elif other.y1 < self.y0:
            return self.y0 - other.y1
        return 0.0
    
    def horizontal_overlap(self, other: 'BoundingBox') -> float:
        """Amount of horizontal overlap (negative if gap)."""
        return min(self.x1, other.x1) - max(self.x0, other.x0)
    
    def vertical_overlap(self, other: 'BoundingBox') -> float:
        """Amount of vertical overlap (negative if gap)."""
        return min(self.y1, other.y1) - max(self.y0, other.y0)
    
    def is_horizontally_aligned(
        self,
        other: 'BoundingBox',
        tolerance: float = 5.0
    ) -> bool:
        """
        Check if two boxes are horizontally aligned (same baseline).
        
        Args:
            other: Box to compare with
            tolerance: Maximum allowed vertical distance
        """
        return abs(self.y0 - other.y0) <= tolerance or abs(self.y1 - other.y1) <= tolerance
    
    def is_vertically_aligned(
        self,
        other: 'BoundingBox',
        tolerance: float = 5.0
    ) -> bool:
        """
        Check if two boxes are vertically aligned (same column).
        
        Args:
            other: Box to compare with
            tolerance: Maximum allowed horizontal distance
        """
        return abs(self.x0 - other.x0) <= tolerance or abs(self.x1 - other.x1) <= tolerance
    
    def is_left_of(self, other: 'BoundingBox', gap_threshold: float = 0) -> bool:
        """Check if this box is to the left of another."""
        return self.x1 + gap_threshold <= other.x0
    
    def is_right_of(self, other: 'BoundingBox', gap_threshold: float = 0) -> bool:
        """Check if this box is to the right of another."""
        return self.x0 >= other.x1 + gap_threshold
    
    def is_above(self, other: 'BoundingBox', gap_threshold: float = 0) -> bool:
        """Check if this box is above another (PDF coordinates)."""
        return self.y0 >= other.y1 + gap_threshold
    
    def is_below(self, other: 'BoundingBox', gap_threshold: float = 0) -> bool:
        """Check if this box is below another (PDF coordinates)."""
        return self.y1 + gap_threshold <= other.y0
    
    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> 'BoundingBox':
        """Create box from list of (x, y) points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(
            x0=min(xs),
            y0=min(ys),
            x1=max(xs),
            y1=max(ys),
        )
    
    @classmethod
    def from_center(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: float
    ) -> 'BoundingBox':
        """Create box from center point and dimensions."""
        half_w = width / 2
        half_h = height / 2
        return cls(
            x0=cx - half_w,
            y0=cy - half_h,
            x1=cx + half_w,
            y1=cy + half_h,
        )


@dataclass
class TextBlock:
    """
    A text element with its bounding box and metadata.
    
    This is the atomic unit of document content - a piece of text
    with its location and attributes.
    
    Attributes:
        text: The text content
        bbox: Bounding box coordinates
        block_type: Classification of the block
        confidence: OCR/extraction confidence (0.0 - 1.0)
        font_name: Font family name if available
        font_size: Font size in points
        is_bold: Whether text is bold
        is_italic: Whether text is italic
        page_number: 0-indexed page number
        line_number: Line number within page
        word_boxes: Individual word bounding boxes if available
        source: Extraction method ('text_layer', 'ocr', 'table')
        metadata: Additional extraction-specific metadata
    """
    text: str
    bbox: BoundingBox
    block_type: BlockType = BlockType.UNKNOWN
    confidence: float = 1.0
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    page_number: int = 0
    line_number: Optional[int] = None
    word_boxes: Optional[List[BoundingBox]] = None
    source: str = 'unknown'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Unique identifier for graph operations
    _id: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if self._id is None:
            # Create deterministic ID from content and position
            self._id = f"block_{self.page_number}_{int(self.bbox.x0)}_{int(self.bbox.y0)}"
    
    @property
    def id(self) -> str:
        """Unique identifier for this block."""
        return self._id
    
    @property
    def is_empty(self) -> bool:
        """Check if text is empty or whitespace only."""
        return not self.text or not self.text.strip()
    
    @property
    def char_count(self) -> int:
        """Number of characters."""
        return len(self.text) if self.text else 0
    
    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split()) if self.text else 0
    
    @property
    def is_numeric(self) -> bool:
        """Check if text is primarily numeric."""
        if not self.text:
            return False
        cleaned = self.text.replace(',', '').replace('.', '').replace('-', '').replace(' ', '')
        return cleaned.isdigit() if cleaned else False
    
    @property
    def is_label_like(self) -> bool:
        """Heuristic check if this looks like a field label."""
        if not self.text:
            return False
        text = self.text.strip()
        # Labels often end with : or have specific patterns
        return (
            text.endswith(':') or
            text.endswith('：') or  # Full-width colon
            (len(text) < 50 and text.isupper()) or
            (self.is_bold and len(text) < 30)
        )
    
    @property
    def normalized_text(self) -> str:
        """Text with normalized whitespace."""
        if not self.text:
            return ''
        return ' '.join(self.text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'bbox': self.bbox.to_dict(),
            'block_type': self.block_type.name,
            'confidence': round(self.confidence, 3),
            'font_name': self.font_name,
            'font_size': self.font_size,
            'is_bold': self.is_bold,
            'is_italic': self.is_italic,
            'page_number': self.page_number,
            'source': self.source,
        }
    
    def distance_to(self, other: 'TextBlock') -> float:
        """Distance to another text block."""
        return self.bbox.distance_to(other.bbox)
    
    def is_same_line(self, other: 'TextBlock', tolerance: float = 5.0) -> bool:
        """Check if two blocks are on the same line."""
        return self.bbox.is_horizontally_aligned(other.bbox, tolerance)
    
    def is_same_column(self, other: 'TextBlock', tolerance: float = 5.0) -> bool:
        """Check if two blocks are in the same column."""
        return self.bbox.is_vertically_aligned(other.bbox, tolerance)
    
    def merge_with(self, other: 'TextBlock', separator: str = ' ') -> 'TextBlock':
        """
        Merge two blocks into one.
        
        The resulting block takes the union of bounding boxes and
        concatenates text. Metadata is merged with self taking precedence.
        """
        # Determine order based on reading direction (left-to-right, top-to-bottom)
        if self.bbox.y0 > other.bbox.y0 or (
            abs(self.bbox.y0 - other.bbox.y0) < 5 and self.bbox.x0 < other.bbox.x0
        ):
            first, second = self, other
        else:
            first, second = other, self
        
        merged_meta = {**other.metadata, **self.metadata}
        
        return TextBlock(
            text=first.text + separator + second.text,
            bbox=self.bbox.union(other.bbox),
            block_type=self.block_type,
            confidence=min(self.confidence, other.confidence),
            font_name=self.font_name or other.font_name,
            font_size=self.font_size or other.font_size,
            is_bold=self.is_bold or other.is_bold,
            is_italic=self.is_italic or other.is_italic,
            page_number=self.page_number,
            source=self.source,
            metadata=merged_meta,
        )


def merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """
    Merge multiple bounding boxes into their union.
    
    Args:
        boxes: List of BoundingBox instances
        
    Returns:
        Single BoundingBox containing all input boxes
        
    Raises:
        ValueError: If boxes list is empty
    """
    if not boxes:
        raise ValueError("Cannot merge empty list of boxes")
    
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def boxes_overlap(box1: BoundingBox, box2: BoundingBox) -> bool:
    """Check if two boxes overlap."""
    return box1.intersects(box2)


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) of two boxes.
    
    IoU is a common metric for measuring overlap between bounding boxes.
    Returns 0 if no intersection, 1 if identical.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
        
    Returns:
        IoU value between 0.0 and 1.0
    """
    intersection = box1.intersection(box2)
    if intersection is None:
        return 0.0
    
    intersection_area = intersection.area
    union_area = box1.area + box2.area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def group_blocks_by_line(
    blocks: List[TextBlock],
    tolerance: float = 5.0
) -> List[List[TextBlock]]:
    """
    Group text blocks into lines based on vertical alignment.
    
    Args:
        blocks: List of TextBlock instances
        tolerance: Maximum vertical distance for same-line grouping
        
    Returns:
        List of lines, each line is a list of TextBlocks sorted left-to-right
    """
    if not blocks:
        return []
    
    # Sort by Y position (top to bottom in PDF coords means high to low y)
    sorted_blocks = sorted(blocks, key=lambda b: (-b.bbox.y0, b.bbox.x0))
    
    lines: List[List[TextBlock]] = []
    current_line: List[TextBlock] = [sorted_blocks[0]]
    current_y = sorted_blocks[0].bbox.y0
    
    for block in sorted_blocks[1:]:
        if abs(block.bbox.y0 - current_y) <= tolerance:
            # Same line
            current_line.append(block)
        else:
            # New line
            # Sort current line left-to-right before adding
            current_line.sort(key=lambda b: b.bbox.x0)
            lines.append(current_line)
            current_line = [block]
            current_y = block.bbox.y0
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda b: b.bbox.x0)
        lines.append(current_line)
    
    return lines


def reading_order_sort(blocks: List[TextBlock]) -> List[TextBlock]:
    """
    Sort blocks in reading order (top-to-bottom, left-to-right).
    
    Uses a two-pass approach:
    1. Group into lines
    2. Sort lines top-to-bottom, blocks within lines left-to-right
    """
    lines = group_blocks_by_line(blocks)
    result = []
    for line in lines:
        result.extend(line)
    return result
