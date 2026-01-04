"""
Spatial Index for Efficient Region Queries

This module provides R-tree based spatial indexing for fast
nearest-neighbor and region queries on text blocks.

Why This Matters:
- Linear search through all blocks is O(n) per query
- With thousands of blocks per page, this becomes a bottleneck
- R-tree provides O(log n) average case for spatial queries

Features:
- Fast nearest-neighbor queries
- Directional queries (find blocks to the right/below/etc.)
- Region queries (find all blocks in an area)
- Dynamic insertion and removal

Implementation Notes:
- Uses rtree library when available for best performance
- Falls back to naive implementation if rtree not installed
- Thread-safe for read operations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional, List, Dict, Tuple, Iterator, Callable, Any, TypeVar, Generic
)
from functools import lru_cache
import heapq

from .box import BoundingBox, TextBlock

logger = logging.getLogger(__name__)

# Try to import rtree for spatial indexing
try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    logger.warning("rtree not available, using naive spatial index (slower)")


class Direction(Enum):
    """Spatial directions for queries."""
    LEFT = auto()
    RIGHT = auto()
    ABOVE = auto()
    BELOW = auto()
    ANY = auto()


@dataclass
class SpatialQuery:
    """
    Query specification for spatial searches.
    
    Attributes:
        origin: Starting point (BoundingBox) for the search
        direction: Which direction to search
        max_distance: Maximum distance from origin (None = unlimited)
        require_alignment: If True, only return aligned results
        alignment_tolerance: Tolerance for alignment checks
        filter_fn: Optional filter function for results
        max_results: Maximum number of results to return
    """
    origin: BoundingBox
    direction: Direction = Direction.ANY
    max_distance: Optional[float] = None
    require_alignment: bool = False
    alignment_tolerance: float = 10.0
    filter_fn: Optional[Callable[[TextBlock], bool]] = None
    max_results: int = 10
    
    def matches_direction(self, target: BoundingBox) -> bool:
        """Check if target is in the specified direction from origin."""
        if self.direction == Direction.ANY:
            return True
        elif self.direction == Direction.LEFT:
            return target.center_x < self.origin.x0
        elif self.direction == Direction.RIGHT:
            return target.x1 > self.origin.x1
        elif self.direction == Direction.ABOVE:
            return target.y0 > self.origin.y1
        elif self.direction == Direction.BELOW:
            return target.y1 < self.origin.y0
        return False
    
    def matches_alignment(self, target: BoundingBox) -> bool:
        """Check if target is aligned with origin."""
        if not self.require_alignment:
            return True
        
        if self.direction in (Direction.LEFT, Direction.RIGHT):
            # Horizontal search: check vertical alignment
            return self.origin.is_horizontally_aligned(target, self.alignment_tolerance)
        elif self.direction in (Direction.ABOVE, Direction.BELOW):
            # Vertical search: check horizontal alignment
            return self.origin.is_vertically_aligned(target, self.alignment_tolerance)
        
        return True


class BaseSpatialIndex:
    """Base class defining spatial index interface."""
    
    def insert(self, block: TextBlock) -> None:
        """Insert a text block into the index."""
        raise NotImplementedError
    
    def remove(self, block: TextBlock) -> bool:
        """Remove a text block from the index."""
        raise NotImplementedError
    
    def query_region(self, region: BoundingBox) -> List[TextBlock]:
        """Find all blocks intersecting a region."""
        raise NotImplementedError
    
    def query_nearest(
        self,
        query: SpatialQuery,
    ) -> List[Tuple[TextBlock, float]]:
        """
        Find nearest blocks matching the query.
        
        Returns list of (block, distance) tuples sorted by distance.
        """
        raise NotImplementedError
    
    def query_direction(
        self,
        origin: BoundingBox,
        direction: Direction,
        max_distance: float = 200,
        require_alignment: bool = True,
        alignment_tolerance: float = 10.0,
    ) -> List[TextBlock]:
        """
        Find blocks in a specific direction from origin.
        
        This is a convenience method that constructs a SpatialQuery.
        """
        query = SpatialQuery(
            origin=origin,
            direction=direction,
            max_distance=max_distance,
            require_alignment=require_alignment,
            alignment_tolerance=alignment_tolerance,
        )
        results = self.query_nearest(query)
        return [block for block, dist in results]


class NaiveSpatialIndex(BaseSpatialIndex):
    """
    Naive O(n) spatial index for when rtree is not available.
    
    This implementation stores all blocks in a list and performs
    linear scans for queries. Suitable for small documents only.
    """
    
    def __init__(self):
        self._blocks: List[TextBlock] = []
        self._block_ids: Dict[str, TextBlock] = {}
    
    @property
    def count(self) -> int:
        """Number of indexed blocks."""
        return len(self._blocks)
    
    def insert(self, block: TextBlock) -> None:
        """Insert a text block into the index."""
        if block.id not in self._block_ids:
            self._blocks.append(block)
            self._block_ids[block.id] = block
    
    def remove(self, block: TextBlock) -> bool:
        """Remove a text block from the index."""
        if block.id in self._block_ids:
            self._blocks = [b for b in self._blocks if b.id != block.id]
            del self._block_ids[block.id]
            return True
        return False
    
    def clear(self) -> None:
        """Remove all blocks from the index."""
        self._blocks.clear()
        self._block_ids.clear()
    
    def query_region(self, region: BoundingBox) -> List[TextBlock]:
        """Find all blocks intersecting a region."""
        return [
            block for block in self._blocks
            if block.bbox.intersects(region)
        ]
    
    def query_point(self, x: float, y: float) -> List[TextBlock]:
        """Find all blocks containing a point."""
        return [
            block for block in self._blocks
            if block.bbox.contains_point(x, y)
        ]
    
    def query_nearest(
        self,
        query: SpatialQuery,
    ) -> List[Tuple[TextBlock, float]]:
        """Find nearest blocks matching the query."""
        candidates: List[Tuple[float, TextBlock]] = []
        
        for block in self._blocks:
            # Skip if doesn't match direction
            if not query.matches_direction(block.bbox):
                continue
            
            # Skip if doesn't match alignment
            if not query.matches_alignment(block.bbox):
                continue
            
            # Skip if custom filter fails
            if query.filter_fn and not query.filter_fn(block):
                continue
            
            # Calculate distance
            distance = query.origin.distance_to(block.bbox)
            
            # Skip if beyond max distance
            if query.max_distance and distance > query.max_distance:
                continue
            
            candidates.append((distance, block))
        
        # Sort by distance and limit results
        candidates.sort(key=lambda x: x[0])
        limited = candidates[:query.max_results]
        
        return [(block, dist) for dist, block in limited]
    
    def get_all(self) -> List[TextBlock]:
        """Get all indexed blocks."""
        return list(self._blocks)


class RTreeSpatialIndex(BaseSpatialIndex):
    """
    R-tree based spatial index for efficient queries.
    
    Uses the rtree library for O(log n) spatial queries.
    Recommended for documents with many text blocks.
    """
    
    def __init__(self):
        # Configure R-tree properties
        p = rtree_index.Property()
        p.dimension = 2
        p.variant = rtree_index.RT_Star  # R*-tree variant for better query performance
        
        self._index = rtree_index.Index(properties=p)
        self._blocks: Dict[int, TextBlock] = {}
        self._id_counter = 0
        self._block_to_id: Dict[str, int] = {}
    
    @property
    def count(self) -> int:
        """Number of indexed blocks."""
        return len(self._blocks)
    
    def _get_next_id(self) -> int:
        """Get next internal ID for rtree."""
        self._id_counter += 1
        return self._id_counter
    
    def insert(self, block: TextBlock) -> None:
        """Insert a text block into the index."""
        if block.id in self._block_to_id:
            return  # Already indexed
        
        internal_id = self._get_next_id()
        
        # rtree uses (x0, y0, x1, y1) format
        bbox_tuple = block.bbox.to_tuple()
        
        self._index.insert(internal_id, bbox_tuple)
        self._blocks[internal_id] = block
        self._block_to_id[block.id] = internal_id
    
    def remove(self, block: TextBlock) -> bool:
        """Remove a text block from the index."""
        if block.id not in self._block_to_id:
            return False
        
        internal_id = self._block_to_id[block.id]
        bbox_tuple = block.bbox.to_tuple()
        
        self._index.delete(internal_id, bbox_tuple)
        del self._blocks[internal_id]
        del self._block_to_id[block.id]
        
        return True
    
    def clear(self) -> None:
        """Remove all blocks from the index."""
        # Recreate index (faster than removing all)
        p = rtree_index.Property()
        p.dimension = 2
        p.variant = rtree_index.RT_Star
        
        self._index = rtree_index.Index(properties=p)
        self._blocks.clear()
        self._block_to_id.clear()
        self._id_counter = 0
    
    def query_region(self, region: BoundingBox) -> List[TextBlock]:
        """Find all blocks intersecting a region."""
        hits = self._index.intersection(region.to_tuple())
        return [self._blocks[h] for h in hits if h in self._blocks]
    
    def query_point(self, x: float, y: float) -> List[TextBlock]:
        """Find all blocks containing a point."""
        # Query with zero-size region
        hits = self._index.intersection((x, y, x, y))
        return [
            self._blocks[h] for h in hits 
            if h in self._blocks and self._blocks[h].bbox.contains_point(x, y)
        ]
    
    def query_nearest(
        self,
        query: SpatialQuery,
    ) -> List[Tuple[TextBlock, float]]:
        """Find nearest blocks matching the query."""
        # Use nearest neighbor query with oversampling
        # (we filter afterwards, so request more than needed)
        oversample = query.max_results * 4
        
        center = query.origin.center
        
        # rtree nearest returns iterator of internal IDs
        try:
            nearest_ids = list(self._index.nearest(
                (center[0], center[1], center[0], center[1]),
                num_results=oversample
            ))
        except Exception:
            # Fallback if nearest fails
            nearest_ids = list(self._blocks.keys())
        
        candidates: List[Tuple[float, TextBlock]] = []
        
        for internal_id in nearest_ids:
            if internal_id not in self._blocks:
                continue
            
            block = self._blocks[internal_id]
            
            # Apply filters
            if not query.matches_direction(block.bbox):
                continue
            if not query.matches_alignment(block.bbox):
                continue
            if query.filter_fn and not query.filter_fn(block):
                continue
            
            distance = query.origin.distance_to(block.bbox)
            
            if query.max_distance and distance > query.max_distance:
                continue
            
            candidates.append((distance, block))
            
            if len(candidates) >= query.max_results:
                break
        
        candidates.sort(key=lambda x: x[0])
        return [(block, dist) for dist, block in candidates[:query.max_results]]
    
    def get_all(self) -> List[TextBlock]:
        """Get all indexed blocks."""
        return list(self._blocks.values())


class SpatialIndex(BaseSpatialIndex):
    """
    Spatial index with automatic backend selection.
    
    Uses R-tree if available, otherwise falls back to naive implementation.
    
    Usage:
        index = SpatialIndex()
        
        # Index all blocks
        for block in text_blocks:
            index.insert(block)
        
        # Find blocks to the right of a label
        results = index.query_direction(
            origin=label.bbox,
            direction=Direction.RIGHT,
            max_distance=200,
            require_alignment=True,
        )
    """
    
    def __init__(self, use_rtree: bool = True):
        """
        Initialize spatial index.
        
        Args:
            use_rtree: Whether to use R-tree (if available)
        """
        if use_rtree and RTREE_AVAILABLE:
            self._impl = RTreeSpatialIndex()
            self._backend = 'rtree'
        else:
            self._impl = NaiveSpatialIndex()
            self._backend = 'naive'
    
    @property
    def backend(self) -> str:
        """Name of the backend implementation."""
        return self._backend
    
    @property
    def count(self) -> int:
        """Number of indexed blocks."""
        return self._impl.count
    
    def insert(self, block: TextBlock) -> None:
        """Insert a text block into the index."""
        self._impl.insert(block)
    
    def insert_many(self, blocks: List[TextBlock]) -> None:
        """Insert multiple blocks."""
        for block in blocks:
            self._impl.insert(block)
    
    def remove(self, block: TextBlock) -> bool:
        """Remove a text block from the index."""
        return self._impl.remove(block)
    
    def clear(self) -> None:
        """Remove all blocks from the index."""
        self._impl.clear()
    
    def query_region(self, region: BoundingBox) -> List[TextBlock]:
        """Find all blocks intersecting a region."""
        return self._impl.query_region(region)
    
    def query_point(self, x: float, y: float) -> List[TextBlock]:
        """Find all blocks containing a point."""
        return self._impl.query_point(x, y)
    
    def query_nearest(
        self,
        query: SpatialQuery,
    ) -> List[Tuple[TextBlock, float]]:
        """Find nearest blocks matching the query."""
        return self._impl.query_nearest(query)
    
    def query_direction(
        self,
        origin: BoundingBox,
        direction: Direction,
        max_distance: float = 200,
        require_alignment: bool = True,
        alignment_tolerance: float = 10.0,
    ) -> List[TextBlock]:
        """Find blocks in a specific direction from origin."""
        return self._impl.query_direction(
            origin, direction, max_distance, require_alignment, alignment_tolerance
        )
    
    def find_right_of(
        self,
        block: TextBlock,
        max_distance: float = 300,
        same_line: bool = True,
    ) -> Optional[TextBlock]:
        """
        Find the nearest block to the right of the given block.
        
        Common use case: finding a value after a label.
        
        Args:
            block: Starting block (usually a label)
            max_distance: Maximum horizontal distance
            same_line: Require vertical alignment
            
        Returns:
            Nearest block to the right, or None
        """
        results = self.query_direction(
            origin=block.bbox,
            direction=Direction.RIGHT,
            max_distance=max_distance,
            require_alignment=same_line,
        )
        return results[0] if results else None
    
    def find_below(
        self,
        block: TextBlock,
        max_distance: float = 100,
        same_column: bool = True,
    ) -> Optional[TextBlock]:
        """
        Find the nearest block below the given block.
        
        Common use case: finding value below header in a table.
        
        Args:
            block: Starting block (usually a header)
            max_distance: Maximum vertical distance
            same_column: Require horizontal alignment
            
        Returns:
            Nearest block below, or None
        """
        results = self.query_direction(
            origin=block.bbox,
            direction=Direction.BELOW,
            max_distance=max_distance,
            require_alignment=same_column,
        )
        return results[0] if results else None
    
    def find_value_for_label(
        self,
        label_block: TextBlock,
        prefer_right: bool = True,
        max_distance: float = 300,
    ) -> Optional[TextBlock]:
        """
        Find the most likely value block for a label.
        
        This implements common document conventions:
        - Values are often to the right of labels
        - Sometimes values are below labels (stacked forms)
        
        Args:
            label_block: The label block to find value for
            prefer_right: Check right side first
            max_distance: Maximum search distance
            
        Returns:
            Best matching value block, or None
        """
        if prefer_right:
            # Try right first
            right_result = self.find_right_of(label_block, max_distance, same_line=True)
            if right_result:
                return right_result
            
            # Try below
            below_result = self.find_below(label_block, max_distance / 3, same_column=True)
            if below_result:
                return below_result
        else:
            # Try below first
            below_result = self.find_below(label_block, max_distance / 3, same_column=True)
            if below_result:
                return below_result
            
            # Try right
            right_result = self.find_right_of(label_block, max_distance, same_line=True)
            if right_result:
                return right_result
        
        return None
    
    def get_all(self) -> List[TextBlock]:
        """Get all indexed blocks."""
        return self._impl.get_all()
