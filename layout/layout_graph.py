"""
Layout Graph for Spatial Reasoning

This module builds a graph representation of document layout where:
- Nodes = text blocks
- Edges = spatial relationships (left_of, below, aligned_with, etc.)

Why a Graph?
- Enables traversal-based field extraction
- Supports complex queries like "largest value near bottom right"
- Can encode document structure (tables, sections, headers)
- Allows probabilistic reasoning about field locations

Relationship Types:
- ADJACENT_LEFT/RIGHT: Horizontally adjacent on same line
- ADJACENT_ABOVE/BELOW: Vertically adjacent in same column
- ALIGNED_HORIZONTAL: Same baseline but not adjacent
- ALIGNED_VERTICAL: Same column but not adjacent
- CONTAINS/CONTAINED_BY: Nesting relationship
- SAME_TABLE_ROW/COLUMN: Table structure relationships
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional, List, Dict, Set, Tuple, Iterator, Callable, Any,
    FrozenSet
)
from collections import defaultdict
import heapq

from .box import BoundingBox, TextBlock, BlockType, group_blocks_by_line
from .spatial_index import SpatialIndex, Direction, SpatialQuery

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of spatial relationships between blocks."""
    
    # Adjacency (directly next to each other)
    ADJACENT_LEFT = auto()      # A is immediately left of B
    ADJACENT_RIGHT = auto()     # A is immediately right of B
    ADJACENT_ABOVE = auto()     # A is immediately above B
    ADJACENT_BELOW = auto()     # A is immediately below B
    
    # Alignment (same row/column but not adjacent)
    ALIGNED_HORIZONTAL = auto() # A and B on same horizontal line
    ALIGNED_VERTICAL = auto()   # A and B in same vertical column
    
    # Containment
    CONTAINS = auto()           # A fully contains B
    CONTAINED_BY = auto()       # A is fully inside B
    
    # Table relationships
    SAME_TABLE_ROW = auto()     # A and B in same table row
    SAME_TABLE_COLUMN = auto()  # A and B in same table column
    TABLE_HEADER_OF = auto()    # A is header of column containing B
    
    # Reading order
    PRECEDES = auto()           # A comes before B in reading order
    FOLLOWS = auto()            # A comes after B in reading order
    
    # Semantic
    LABEL_VALUE = auto()        # A is label for value B


class RegionType(Enum):
    """Types of document regions."""
    HEADER = auto()         # Page/document header
    FOOTER = auto()         # Page/document footer
    BODY = auto()           # Main content area
    SIDEBAR = auto()        # Side content
    TABLE = auto()          # Table region
    FIGURE = auto()         # Image/figure region
    FORM_FIELD = auto()     # Form input area
    SIGNATURE = auto()      # Signature area
    LOGO = auto()           # Logo area


@dataclass
class SpatialRelation:
    """
    A spatial relationship between two text blocks.
    
    Attributes:
        source_id: ID of the source block
        target_id: ID of the target block
        relation_type: Type of relationship
        confidence: Confidence in this relationship (0.0 - 1.0)
        distance: Physical distance between blocks
        metadata: Additional relationship-specific data
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    confidence: float = 1.0
    distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type.name,
            'confidence': round(self.confidence, 3),
            'distance': round(self.distance, 2),
            'metadata': self.metadata,
        }


@dataclass
class LayoutRegion:
    """
    A defined region of the document.
    
    Regions segment the document into logical areas like header,
    footer, tables, etc. for more targeted processing.
    """
    region_type: RegionType
    bbox: BoundingBox
    page_number: int
    confidence: float = 1.0
    blocks: List[str] = field(default_factory=list)  # Block IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate region ID."""
        return f"region_{self.region_type.name}_{self.page_number}_{int(self.bbox.x0)}"
    
    def contains_block(self, block: TextBlock) -> bool:
        """Check if block is inside this region."""
        return self.bbox.contains_box(block.bbox)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'region_type': self.region_type.name,
            'bbox': self.bbox.to_dict(),
            'page_number': self.page_number,
            'confidence': round(self.confidence, 3),
            'block_count': len(self.blocks),
        }


class LayoutGraph:
    """
    Graph representation of document layout.
    
    The graph enables spatial reasoning about document structure.
    Each text block is a node, and edges represent spatial relationships.
    
    Features:
    - Build graph from text blocks automatically
    - Query for specific relationship patterns
    - Detect document regions (tables, headers, etc.)
    - Support for label-value extraction patterns
    
    Usage:
        graph = LayoutGraph()
        graph.add_blocks(text_blocks)
        graph.build_relationships()
        
        # Find values for labels
        label = graph.find_block_by_text("Invoice Number:")
        value = graph.find_value_for_label(label)
        
        # Query graph
        adjacent = graph.get_adjacent(block, Direction.RIGHT)
        aligned = graph.get_aligned_horizontal(block)
    """
    
    def __init__(
        self,
        adjacency_threshold: float = 50.0,
        alignment_tolerance: float = 10.0,
    ):
        """
        Initialize layout graph.
        
        Args:
            adjacency_threshold: Max distance for adjacency relationships
            alignment_tolerance: Tolerance for alignment detection
        """
        self.adjacency_threshold = adjacency_threshold
        self.alignment_tolerance = alignment_tolerance
        
        # Node storage
        self._blocks: Dict[str, TextBlock] = {}
        
        # Edge storage (adjacency list)
        self._outgoing: Dict[str, List[SpatialRelation]] = defaultdict(list)
        self._incoming: Dict[str, List[SpatialRelation]] = defaultdict(list)
        
        # Spatial index for efficient queries
        self._spatial_index = SpatialIndex()
        
        # Detected regions
        self._regions: List[LayoutRegion] = []
        
        # Page dimensions (needed for some calculations)
        self._page_dimensions: Dict[int, Tuple[float, float]] = {}
    
    @property
    def block_count(self) -> int:
        """Number of blocks in the graph."""
        return len(self._blocks)
    
    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return sum(len(edges) for edges in self._outgoing.values())
    
    @property
    def blocks(self) -> List[TextBlock]:
        """All blocks in the graph."""
        return list(self._blocks.values())
    
    @property
    def regions(self) -> List[LayoutRegion]:
        """All detected regions."""
        return list(self._regions)
    
    def set_page_dimensions(self, page_number: int, width: float, height: float):
        """Set dimensions for a page."""
        self._page_dimensions[page_number] = (width, height)
    
    def add_block(self, block: TextBlock) -> None:
        """Add a single block to the graph."""
        if block.id not in self._blocks:
            self._blocks[block.id] = block
            self._spatial_index.insert(block)
    
    def add_blocks(self, blocks: List[TextBlock]) -> None:
        """Add multiple blocks to the graph."""
        for block in blocks:
            self.add_block(block)
    
    def get_block(self, block_id: str) -> Optional[TextBlock]:
        """Get a block by ID."""
        return self._blocks.get(block_id)
    
    def find_block_by_text(
        self,
        text: str,
        exact: bool = False,
        case_sensitive: bool = False,
    ) -> Optional[TextBlock]:
        """
        Find a block containing specific text.
        
        Args:
            text: Text to search for
            exact: If True, require exact match
            case_sensitive: If True, match case
            
        Returns:
            First matching block or None
        """
        search_text = text if case_sensitive else text.lower()
        
        for block in self._blocks.values():
            block_text = block.text if case_sensitive else block.text.lower()
            
            if exact:
                if block_text.strip() == search_text.strip():
                    return block
            else:
                if search_text in block_text:
                    return block
        
        return None
    
    def find_blocks_by_text(
        self,
        text: str,
        exact: bool = False,
        case_sensitive: bool = False,
    ) -> List[TextBlock]:
        """Find all blocks containing specific text."""
        search_text = text if case_sensitive else text.lower()
        results = []
        
        for block in self._blocks.values():
            block_text = block.text if case_sensitive else block.text.lower()
            
            if exact:
                if block_text.strip() == search_text.strip():
                    results.append(block)
            else:
                if search_text in block_text:
                    results.append(block)
        
        return results
    
    def add_relation(self, relation: SpatialRelation) -> None:
        """Add a relationship between blocks."""
        self._outgoing[relation.source_id].append(relation)
        self._incoming[relation.target_id].append(relation)
    
    def get_outgoing(
        self,
        block_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[SpatialRelation]:
        """Get outgoing relationships from a block."""
        relations = self._outgoing.get(block_id, [])
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        return relations
    
    def get_incoming(
        self,
        block_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> List[SpatialRelation]:
        """Get incoming relationships to a block."""
        relations = self._incoming.get(block_id, [])
        if relation_type:
            relations = [r for r in relations if r.relation_type == relation_type]
        return relations
    
    def get_related_blocks(
        self,
        block: TextBlock,
        relation_type: RelationType,
        direction: str = 'outgoing',
    ) -> List[TextBlock]:
        """
        Get blocks related to the given block.
        
        Args:
            block: Source block
            relation_type: Type of relationship
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of related blocks
        """
        block_ids: Set[str] = set()
        
        if direction in ('outgoing', 'both'):
            for rel in self.get_outgoing(block.id, relation_type):
                block_ids.add(rel.target_id)
        
        if direction in ('incoming', 'both'):
            for rel in self.get_incoming(block.id, relation_type):
                block_ids.add(rel.source_id)
        
        return [self._blocks[bid] for bid in block_ids if bid in self._blocks]
    
    def build_relationships(self) -> None:
        """
        Build all spatial relationships between blocks.
        
        This analyzes the spatial arrangement of blocks and creates
        edges for adjacency, alignment, containment, and reading order.
        """
        blocks = list(self._blocks.values())
        
        if not blocks:
            return
        
        logger.debug(f"Building relationships for {len(blocks)} blocks")
        
        # Group blocks by page
        pages: Dict[int, List[TextBlock]] = defaultdict(list)
        for block in blocks:
            pages[block.page_number].append(block)
        
        # Process each page
        for page_num, page_blocks in pages.items():
            self._build_page_relationships(page_blocks)
        
        # Build cross-page relationships if needed
        self._build_reading_order(blocks)
        
        logger.debug(f"Built {self.edge_count} relationships")
    
    def _build_page_relationships(self, blocks: List[TextBlock]) -> None:
        """Build relationships for blocks on a single page."""
        
        # Group into lines
        lines = group_blocks_by_line(blocks, self.alignment_tolerance)
        
        # Build horizontal adjacency within lines
        for line in lines:
            for i in range(len(line) - 1):
                left_block = line[i]
                right_block = line[i + 1]
                
                distance = right_block.bbox.x0 - left_block.bbox.x1
                
                if distance <= self.adjacency_threshold:
                    # Adjacent
                    self.add_relation(SpatialRelation(
                        source_id=left_block.id,
                        target_id=right_block.id,
                        relation_type=RelationType.ADJACENT_RIGHT,
                        distance=distance,
                    ))
                    self.add_relation(SpatialRelation(
                        source_id=right_block.id,
                        target_id=left_block.id,
                        relation_type=RelationType.ADJACENT_LEFT,
                        distance=distance,
                    ))
                
                # Aligned (even if not adjacent)
                self.add_relation(SpatialRelation(
                    source_id=left_block.id,
                    target_id=right_block.id,
                    relation_type=RelationType.ALIGNED_HORIZONTAL,
                    distance=distance,
                ))
        
        # Build vertical relationships (between lines)
        for i in range(len(lines) - 1):
            upper_line = lines[i]
            lower_line = lines[i + 1]
            
            for upper_block in upper_line:
                for lower_block in lower_line:
                    # Check vertical alignment
                    h_overlap = upper_block.bbox.horizontal_overlap(lower_block.bbox)
                    
                    if h_overlap > 0:
                        v_distance = upper_block.bbox.y0 - lower_block.bbox.y1
                        
                        if v_distance <= self.adjacency_threshold:
                            self.add_relation(SpatialRelation(
                                source_id=upper_block.id,
                                target_id=lower_block.id,
                                relation_type=RelationType.ADJACENT_BELOW,
                                distance=v_distance,
                            ))
                            self.add_relation(SpatialRelation(
                                source_id=lower_block.id,
                                target_id=upper_block.id,
                                relation_type=RelationType.ADJACENT_ABOVE,
                                distance=v_distance,
                            ))
                        
                        # Vertical alignment
                        if abs(upper_block.bbox.center_x - lower_block.bbox.center_x) <= self.alignment_tolerance:
                            self.add_relation(SpatialRelation(
                                source_id=upper_block.id,
                                target_id=lower_block.id,
                                relation_type=RelationType.ALIGNED_VERTICAL,
                            ))
        
        # Detect label-value relationships
        self._detect_label_value_pairs(blocks)
    
    def _detect_label_value_pairs(self, blocks: List[TextBlock]) -> None:
        """Detect label-value relationships."""
        for block in blocks:
            if block.is_label_like:
                # Find potential value to the right
                value_block = self._spatial_index.find_right_of(
                    block,
                    max_distance=300,
                    same_line=True,
                )
                
                if value_block and not value_block.is_label_like:
                    self.add_relation(SpatialRelation(
                        source_id=block.id,
                        target_id=value_block.id,
                        relation_type=RelationType.LABEL_VALUE,
                        confidence=0.8,
                    ))
    
    def _build_reading_order(self, blocks: List[TextBlock]) -> None:
        """Build reading order relationships."""
        # Sort by page, then by position
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (b.page_number, -b.bbox.y0, b.bbox.x0)
        )
        
        for i in range(len(sorted_blocks) - 1):
            current = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]
            
            self.add_relation(SpatialRelation(
                source_id=current.id,
                target_id=next_block.id,
                relation_type=RelationType.PRECEDES,
            ))
    
    def detect_regions(self) -> List[LayoutRegion]:
        """
        Detect document regions (header, footer, tables, etc.).
        
        Returns:
            List of detected regions
        """
        self._regions.clear()
        
        # Group blocks by page
        pages: Dict[int, List[TextBlock]] = defaultdict(list)
        for block in self._blocks.values():
            pages[block.page_number].append(block)
        
        for page_num, page_blocks in pages.items():
            page_regions = self._detect_page_regions(page_num, page_blocks)
            self._regions.extend(page_regions)
        
        return self._regions
    
    def _detect_page_regions(
        self,
        page_number: int,
        blocks: List[TextBlock],
    ) -> List[LayoutRegion]:
        """Detect regions on a single page."""
        regions = []
        
        if not blocks:
            return regions
        
        # Get page dimensions
        page_dims = self._page_dimensions.get(page_number)
        if not page_dims:
            # Estimate from block positions
            max_x = max(b.bbox.x1 for b in blocks)
            max_y = max(b.bbox.y1 for b in blocks)
            page_dims = (max_x + 50, max_y + 50)
        
        page_width, page_height = page_dims
        
        # Detect header region (top 10% of page)
        header_threshold = page_height * 0.9
        header_blocks = [b for b in blocks if b.bbox.y0 > header_threshold]
        
        if header_blocks:
            header_bbox = BoundingBox(
                x0=0,
                y0=header_threshold,
                x1=page_width,
                y1=page_height,
            )
            regions.append(LayoutRegion(
                region_type=RegionType.HEADER,
                bbox=header_bbox,
                page_number=page_number,
                blocks=[b.id for b in header_blocks],
            ))
        
        # Detect footer region (bottom 10% of page)
        footer_threshold = page_height * 0.1
        footer_blocks = [b for b in blocks if b.bbox.y1 < footer_threshold]
        
        if footer_blocks:
            footer_bbox = BoundingBox(
                x0=0,
                y0=0,
                x1=page_width,
                y1=footer_threshold,
            )
            regions.append(LayoutRegion(
                region_type=RegionType.FOOTER,
                bbox=footer_bbox,
                page_number=page_number,
                blocks=[b.id for b in footer_blocks],
            ))
        
        # Detect table regions (aligned blocks in grid pattern)
        table_regions = self._detect_table_regions(blocks)
        regions.extend(table_regions)
        
        return regions
    
    def _detect_table_regions(self, blocks: List[TextBlock]) -> List[LayoutRegion]:
        """Detect table regions based on grid patterns."""
        regions = []
        
        # Simple heuristic: look for blocks with similar x-coordinates (columns)
        # and blocks with similar y-coordinates (rows)
        
        if len(blocks) < 4:  # Need at least 2x2 for a table
            return regions
        
        # Group by approximate x-coordinate (columns)
        x_groups: Dict[int, List[TextBlock]] = defaultdict(list)
        for block in blocks:
            x_key = int(block.bbox.x0 / 20) * 20  # Round to nearest 20 points
            x_groups[x_key].append(block)
        
        # Find groups that could be table columns (at least 3 blocks)
        potential_columns = [
            group for group in x_groups.values()
            if len(group) >= 3
        ]
        
        if len(potential_columns) >= 2:
            # This might be a table - combine all blocks
            table_blocks = [b for col in potential_columns for b in col]
            
            if len(table_blocks) >= 6:  # At least 2 rows x 3 columns or similar
                # Create bounding box for table
                all_bboxes = [b.bbox for b in table_blocks]
                table_bbox = BoundingBox(
                    x0=min(b.x0 for b in all_bboxes),
                    y0=min(b.y0 for b in all_bboxes),
                    x1=max(b.x1 for b in all_bboxes),
                    y1=max(b.y1 for b in all_bboxes),
                )
                
                page_num = table_blocks[0].page_number
                
                regions.append(LayoutRegion(
                    region_type=RegionType.TABLE,
                    bbox=table_bbox,
                    page_number=page_num,
                    blocks=[b.id for b in table_blocks],
                    confidence=0.7,  # Lower confidence for heuristic detection
                ))
        
        return regions
    
    def find_value_for_label(
        self,
        label: TextBlock,
        prefer_right: bool = True,
    ) -> Optional[TextBlock]:
        """
        Find the most likely value for a given label.
        
        Uses graph relationships and spatial reasoning.
        
        Args:
            label: The label block
            prefer_right: Whether to prefer right-side values
            
        Returns:
            The value block, or None
        """
        # First check explicit label-value relationships
        label_value_rels = self.get_outgoing(label.id, RelationType.LABEL_VALUE)
        if label_value_rels:
            best_rel = max(label_value_rels, key=lambda r: r.confidence)
            return self.get_block(best_rel.target_id)
        
        # Fall back to adjacency
        if prefer_right:
            # Try right adjacency
            right_rels = self.get_outgoing(label.id, RelationType.ADJACENT_RIGHT)
            if right_rels:
                return self.get_block(right_rels[0].target_id)
            
            # Try below adjacency
            below_rels = self.get_outgoing(label.id, RelationType.ADJACENT_BELOW)
            if below_rels:
                return self.get_block(below_rels[0].target_id)
        else:
            # Try below first
            below_rels = self.get_outgoing(label.id, RelationType.ADJACENT_BELOW)
            if below_rels:
                return self.get_block(below_rels[0].target_id)
            
            right_rels = self.get_outgoing(label.id, RelationType.ADJACENT_RIGHT)
            if right_rels:
                return self.get_block(right_rels[0].target_id)
        
        # Last resort: use spatial index
        return self._spatial_index.find_value_for_label(label, prefer_right)
    
    def get_blocks_in_region(self, region: LayoutRegion) -> List[TextBlock]:
        """Get all blocks within a region."""
        return [
            self._blocks[bid] for bid in region.blocks
            if bid in self._blocks
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary for serialization."""
        return {
            'blocks': [b.to_dict() for b in self._blocks.values()],
            'relations': [
                rel.to_dict()
                for rels in self._outgoing.values()
                for rel in rels
            ],
            'regions': [r.to_dict() for r in self._regions],
            'statistics': {
                'block_count': self.block_count,
                'edge_count': self.edge_count,
                'region_count': len(self._regions),
            }
        }
