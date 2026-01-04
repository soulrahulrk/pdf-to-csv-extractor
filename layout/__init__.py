"""
Layout Intelligence Package

Provides spatial reasoning capabilities for document understanding.
This package enables layout-aware field extraction by building a graph
of text elements with their spatial relationships.

Key Components:
- BoundingBox: Geometric representation of text regions
- TextBlock: Text content with position and metadata
- SpatialIndex: R-tree based spatial querying
- LayoutGraph: Graph structure for spatial relationships
- LayoutAnalyzer: High-level layout analysis orchestration

Spatial Relationships Supported:
- left_of / right_of: Horizontal adjacency
- above / below: Vertical adjacency
- aligned_with: Baseline or edge alignment
- contains / contained_by: Nesting relationships
- near: Proximity within threshold

Usage:
    from layout import LayoutAnalyzer, BoundingBox
    
    analyzer = LayoutAnalyzer()
    layout = analyzer.analyze_page(pdf_path, page_num=0)
    
    # Find value to the right of a label
    label = layout.find_text("Invoice Number:")
    value = layout.find_right_of(label, max_distance=200)
"""

from .box import (
    BoundingBox,
    TextBlock,
    BlockType,
    merge_boxes,
    boxes_overlap,
    calculate_iou,
)
from .spatial_index import (
    SpatialIndex,
    SpatialQuery,
    Direction,
)
from .layout_graph import (
    LayoutGraph,
    SpatialRelation,
    RelationType,
    LayoutRegion,
    RegionType,
)
from .analyzer import (
    LayoutAnalyzer,
    PageLayout,
    DocumentLayout,
    LayoutExtractionResult,
)

__all__ = [
    # Geometric primitives
    'BoundingBox',
    'TextBlock', 
    'BlockType',
    'merge_boxes',
    'boxes_overlap',
    'calculate_iou',
    
    # Spatial indexing
    'SpatialIndex',
    'SpatialQuery',
    'Direction',
    
    # Layout graph
    'LayoutGraph',
    'SpatialRelation',
    'RelationType',
    'LayoutRegion',
    'RegionType',
    
    # Analysis
    'LayoutAnalyzer',
    'PageLayout',
    'DocumentLayout',
    'LayoutExtractionResult',
]
