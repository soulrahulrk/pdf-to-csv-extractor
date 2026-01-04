"""
Layout Analyzer - High-Level Layout Analysis

This module provides the main entry point for layout analysis.
It orchestrates the extraction of text with bounding boxes,
builds the layout graph, and provides querying capabilities.

Usage:
    from layout import LayoutAnalyzer
    
    analyzer = LayoutAnalyzer()
    doc_layout = analyzer.analyze_document(pdf_path)
    
    # Find field values using spatial reasoning
    for page in doc_layout.pages:
        label = page.find_text("Invoice Number:")
        if label:
            value = page.find_value_for_label(label)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Optional, List, Dict, Tuple, Iterator, Any, Union
)
from enum import Enum, auto

from .box import BoundingBox, TextBlock, BlockType, group_blocks_by_line
from .spatial_index import SpatialIndex, Direction
from .layout_graph import LayoutGraph, LayoutRegion, RegionType, SpatialRelation

logger = logging.getLogger(__name__)

# Try to import PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class ExtractionBackend(Enum):
    """Available PDF extraction backends."""
    PDFPLUMBER = auto()
    PYMUPDF = auto()
    AUTO = auto()


@dataclass
class LayoutExtractionResult:
    """
    Result of layout-aware text extraction.
    
    Contains the extracted text blocks with their positions,
    spatial relationships, and confidence information.
    """
    blocks: List[TextBlock]
    page_number: int
    page_width: float
    page_height: float
    backend: str
    warnings: List[str] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Get concatenated text in reading order."""
        sorted_blocks = sorted(
            self.blocks,
            key=lambda b: (-b.bbox.y0, b.bbox.x0)
        )
        return '\n'.join(b.text for b in sorted_blocks if b.text.strip())
    
    @property
    def block_count(self) -> int:
        """Number of extracted blocks."""
        return len(self.blocks)
    
    @property
    def has_text(self) -> bool:
        """Whether any text was extracted."""
        return any(b.text.strip() for b in self.blocks)


@dataclass
class PageLayout:
    """
    Layout analysis results for a single page.
    
    Provides spatial querying and field extraction for one page.
    """
    extraction: LayoutExtractionResult
    graph: LayoutGraph
    regions: List[LayoutRegion]
    _spatial_index: SpatialIndex = field(default_factory=SpatialIndex)
    
    def __post_init__(self):
        """Build spatial index after initialization."""
        if self.extraction.blocks:
            self._spatial_index.insert_many(self.extraction.blocks)
    
    @property
    def page_number(self) -> int:
        return self.extraction.page_number
    
    @property
    def width(self) -> float:
        return self.extraction.page_width
    
    @property
    def height(self) -> float:
        return self.extraction.page_height
    
    @property
    def blocks(self) -> List[TextBlock]:
        return self.extraction.blocks
    
    @property
    def text(self) -> str:
        return self.extraction.text
    
    def find_text(
        self,
        text: str,
        exact: bool = False,
        case_sensitive: bool = False,
    ) -> Optional[TextBlock]:
        """Find a block containing specific text."""
        return self.graph.find_block_by_text(text, exact, case_sensitive)
    
    def find_all_text(
        self,
        text: str,
        exact: bool = False,
        case_sensitive: bool = False,
    ) -> List[TextBlock]:
        """Find all blocks containing specific text."""
        return self.graph.find_blocks_by_text(text, exact, case_sensitive)
    
    def find_value_for_label(
        self,
        label: TextBlock,
        prefer_right: bool = True,
    ) -> Optional[TextBlock]:
        """Find the value associated with a label."""
        return self.graph.find_value_for_label(label, prefer_right)
    
    def find_right_of(
        self,
        block: TextBlock,
        max_distance: float = 300,
        same_line: bool = True,
    ) -> Optional[TextBlock]:
        """Find the nearest block to the right."""
        return self._spatial_index.find_right_of(block, max_distance, same_line)
    
    def find_below(
        self,
        block: TextBlock,
        max_distance: float = 100,
        same_column: bool = True,
    ) -> Optional[TextBlock]:
        """Find the nearest block below."""
        return self._spatial_index.find_below(block, max_distance, same_column)
    
    def find_in_region(
        self,
        region_type: RegionType,
    ) -> List[TextBlock]:
        """Find all blocks in regions of a specific type."""
        blocks = []
        for region in self.regions:
            if region.region_type == region_type:
                blocks.extend(self.graph.get_blocks_in_region(region))
        return blocks
    
    def query_region(self, bbox: BoundingBox) -> List[TextBlock]:
        """Find all blocks in a bounding box region."""
        return self._spatial_index.query_region(bbox)
    
    def get_lines(self) -> List[List[TextBlock]]:
        """Get blocks grouped by lines."""
        return group_blocks_by_line(self.blocks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export page layout as dictionary."""
        return {
            'page_number': self.page_number,
            'width': self.width,
            'height': self.height,
            'block_count': len(self.blocks),
            'regions': [r.to_dict() for r in self.regions],
            'blocks': [b.to_dict() for b in self.blocks],
        }


@dataclass
class DocumentLayout:
    """
    Layout analysis results for an entire document.
    
    Contains per-page layouts and document-wide analysis.
    """
    pages: List[PageLayout]
    source_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def page_count(self) -> int:
        """Number of pages."""
        return len(self.pages)
    
    @property
    def total_blocks(self) -> int:
        """Total number of text blocks across all pages."""
        return sum(len(p.blocks) for p in self.pages)
    
    @property
    def text(self) -> str:
        """Get all text from document."""
        return '\n\n'.join(p.text for p in self.pages)
    
    def get_page(self, page_number: int) -> Optional[PageLayout]:
        """Get layout for a specific page (0-indexed)."""
        if 0 <= page_number < len(self.pages):
            return self.pages[page_number]
        return None
    
    def find_text(
        self,
        text: str,
        exact: bool = False,
        case_sensitive: bool = False,
    ) -> List[Tuple[int, TextBlock]]:
        """
        Find text across all pages.
        
        Returns list of (page_number, block) tuples.
        """
        results = []
        for page in self.pages:
            blocks = page.find_all_text(text, exact, case_sensitive)
            for block in blocks:
                results.append((page.page_number, block))
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """Export document layout as dictionary."""
        return {
            'source_path': str(self.source_path) if self.source_path else None,
            'page_count': self.page_count,
            'total_blocks': self.total_blocks,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'pages': [p.to_dict() for p in self.pages],
        }


class LayoutAnalyzer:
    """
    High-level layout analyzer for PDF documents.
    
    Extracts text with bounding boxes, builds layout graphs,
    and provides spatial querying capabilities.
    
    Features:
    - Multiple backend support (pdfplumber, PyMuPDF)
    - Automatic backend selection
    - Font and style information extraction
    - Block type classification
    - Region detection (header, footer, tables)
    - Spatial relationship building
    
    Usage:
        analyzer = LayoutAnalyzer()
        
        # Analyze entire document
        layout = analyzer.analyze_document(pdf_path)
        
        # Find fields using spatial reasoning
        for page in layout.pages:
            label = page.find_text("Total:")
            if label:
                value = page.find_value_for_label(label)
                print(f"Total: {value.text if value else 'Not found'}")
        
        # Analyze single page
        page_layout = analyzer.analyze_page(pdf_path, page_num=0)
    """
    
    def __init__(
        self,
        backend: ExtractionBackend = ExtractionBackend.AUTO,
        adjacency_threshold: float = 50.0,
        alignment_tolerance: float = 10.0,
        detect_regions: bool = True,
    ):
        """
        Initialize layout analyzer.
        
        Args:
            backend: Which PDF library to use
            adjacency_threshold: Max distance for adjacency relationships
            alignment_tolerance: Tolerance for alignment detection
            detect_regions: Whether to detect document regions
        """
        self.adjacency_threshold = adjacency_threshold
        self.alignment_tolerance = alignment_tolerance
        self.detect_regions = detect_regions
        
        # Select backend
        if backend == ExtractionBackend.AUTO:
            if PDFPLUMBER_AVAILABLE:
                self._backend = ExtractionBackend.PDFPLUMBER
            elif PYMUPDF_AVAILABLE:
                self._backend = ExtractionBackend.PYMUPDF
            else:
                raise RuntimeError(
                    "No PDF library available. Install pdfplumber or PyMuPDF."
                )
        else:
            self._backend = backend
            # Verify backend is available
            if backend == ExtractionBackend.PDFPLUMBER and not PDFPLUMBER_AVAILABLE:
                raise RuntimeError("pdfplumber not available")
            if backend == ExtractionBackend.PYMUPDF and not PYMUPDF_AVAILABLE:
                raise RuntimeError("PyMuPDF not available")
    
    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend.name
    
    def analyze_document(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
    ) -> DocumentLayout:
        """
        Analyze layout of entire document.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to analyze (0-indexed), or None for all
            
        Returns:
            DocumentLayout with per-page analysis
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        page_layouts = []
        warnings = []
        metadata = {}
        
        if self._backend == ExtractionBackend.PDFPLUMBER:
            page_layouts, warnings, metadata = self._analyze_with_pdfplumber(
                pdf_path, pages
            )
        elif self._backend == ExtractionBackend.PYMUPDF:
            page_layouts, warnings, metadata = self._analyze_with_pymupdf(
                pdf_path, pages
            )
        
        return DocumentLayout(
            pages=page_layouts,
            source_path=pdf_path,
            warnings=warnings,
            metadata=metadata,
        )
    
    def analyze_page(
        self,
        pdf_path: Union[str, Path],
        page_num: int = 0,
    ) -> PageLayout:
        """
        Analyze layout of a single page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            
        Returns:
            PageLayout for the specified page
        """
        doc_layout = self.analyze_document(pdf_path, pages=[page_num])
        
        if not doc_layout.pages:
            raise ValueError(f"Page {page_num} not found in document")
        
        return doc_layout.pages[0]
    
    def _analyze_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
    ) -> Tuple[List[PageLayout], List[str], Dict[str, Any]]:
        """Analyze using pdfplumber."""
        page_layouts = []
        warnings = []
        metadata = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                'page_count': len(pdf.pages),
                'metadata': pdf.metadata or {},
            }
            
            page_nums = pages if pages else range(len(pdf.pages))
            
            for page_num in page_nums:
                if page_num >= len(pdf.pages):
                    warnings.append(f"Page {page_num} not found")
                    continue
                
                page = pdf.pages[page_num]
                page_layout = self._extract_page_pdfplumber(page, page_num)
                page_layouts.append(page_layout)
        
        return page_layouts, warnings, metadata
    
    def _extract_page_pdfplumber(
        self,
        page,
        page_num: int,
    ) -> PageLayout:
        """Extract layout from a pdfplumber page."""
        blocks = []
        warnings = []
        
        page_width = page.width
        page_height = page.height
        
        # Extract words with their bounding boxes
        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3,
            extra_attrs=['fontname', 'size'],
        )
        
        # Group words into blocks by proximity
        # For now, treat each word as a block (can be improved)
        for i, word in enumerate(words):
            if not word.get('text', '').strip():
                continue
            
            bbox = BoundingBox(
                x0=word['x0'],
                y0=page_height - word['bottom'],  # Convert to PDF coords
                x1=word['x1'],
                y1=page_height - word['top'],
            )
            
            # Detect font attributes
            font_name = word.get('fontname', '')
            font_size = word.get('size', 12)
            is_bold = 'bold' in font_name.lower() or 'heavy' in font_name.lower()
            is_italic = 'italic' in font_name.lower() or 'oblique' in font_name.lower()
            
            block = TextBlock(
                text=word['text'],
                bbox=bbox,
                confidence=1.0,  # Text layer is high confidence
                font_name=font_name,
                font_size=font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                page_number=page_num,
                source='pdfplumber',
            )
            
            blocks.append(block)
        
        # Merge words into lines
        blocks = self._merge_into_lines(blocks)
        
        # Create extraction result
        extraction = LayoutExtractionResult(
            blocks=blocks,
            page_number=page_num,
            page_width=page_width,
            page_height=page_height,
            backend='pdfplumber',
            warnings=warnings,
        )
        
        # Build layout graph
        graph = LayoutGraph(
            adjacency_threshold=self.adjacency_threshold,
            alignment_tolerance=self.alignment_tolerance,
        )
        graph.add_blocks(blocks)
        graph.set_page_dimensions(page_num, page_width, page_height)
        graph.build_relationships()
        
        # Detect regions
        regions = []
        if self.detect_regions:
            regions = graph.detect_regions()
        
        return PageLayout(
            extraction=extraction,
            graph=graph,
            regions=regions,
        )
    
    def _analyze_with_pymupdf(
        self,
        pdf_path: Path,
        pages: Optional[List[int]],
    ) -> Tuple[List[PageLayout], List[str], Dict[str, Any]]:
        """Analyze using PyMuPDF."""
        page_layouts = []
        warnings = []
        metadata = {}
        
        doc = fitz.open(pdf_path)
        
        try:
            metadata = {
                'page_count': len(doc),
                'metadata': doc.metadata or {},
            }
            
            page_nums = pages if pages else range(len(doc))
            
            for page_num in page_nums:
                if page_num >= len(doc):
                    warnings.append(f"Page {page_num} not found")
                    continue
                
                page = doc[page_num]
                page_layout = self._extract_page_pymupdf(page, page_num)
                page_layouts.append(page_layout)
        finally:
            doc.close()
        
        return page_layouts, warnings, metadata
    
    def _extract_page_pymupdf(
        self,
        page,
        page_num: int,
    ) -> PageLayout:
        """Extract layout from a PyMuPDF page."""
        blocks = []
        warnings = []
        
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Get text blocks with position info
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        for block in text_dict.get('blocks', []):
            if block['type'] != 0:  # Skip non-text blocks
                continue
            
            for line in block.get('lines', []):
                line_text_parts = []
                line_boxes = []
                font_name = None
                font_size = None
                is_bold = False
                is_italic = False
                
                for span in line.get('spans', []):
                    text = span.get('text', '')
                    if not text.strip():
                        continue
                    
                    line_text_parts.append(text)
                    
                    span_bbox = span.get('bbox', (0, 0, 0, 0))
                    line_boxes.append(BoundingBox(
                        x0=span_bbox[0],
                        y0=page_height - span_bbox[3],  # Convert to PDF coords
                        x1=span_bbox[2],
                        y1=page_height - span_bbox[1],
                    ))
                    
                    # Capture font info from first span
                    if font_name is None:
                        font_name = span.get('font', '')
                        font_size = span.get('size', 12)
                        flags = span.get('flags', 0)
                        is_bold = bool(flags & 2 ** 4)  # Bold flag
                        is_italic = bool(flags & 2 ** 1)  # Italic flag
                
                if line_text_parts and line_boxes:
                    # Merge line boxes
                    merged_bbox = line_boxes[0]
                    for box in line_boxes[1:]:
                        merged_bbox = merged_bbox.union(box)
                    
                    line_text = ' '.join(line_text_parts)
                    
                    text_block = TextBlock(
                        text=line_text,
                        bbox=merged_bbox,
                        confidence=1.0,
                        font_name=font_name,
                        font_size=font_size,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        page_number=page_num,
                        source='pymupdf',
                        word_boxes=line_boxes if len(line_boxes) > 1 else None,
                    )
                    
                    blocks.append(text_block)
        
        # Create extraction result
        extraction = LayoutExtractionResult(
            blocks=blocks,
            page_number=page_num,
            page_width=page_width,
            page_height=page_height,
            backend='pymupdf',
            warnings=warnings,
        )
        
        # Build layout graph
        graph = LayoutGraph(
            adjacency_threshold=self.adjacency_threshold,
            alignment_tolerance=self.alignment_tolerance,
        )
        graph.add_blocks(blocks)
        graph.set_page_dimensions(page_num, page_width, page_height)
        graph.build_relationships()
        
        # Detect regions
        regions = []
        if self.detect_regions:
            regions = graph.detect_regions()
        
        return PageLayout(
            extraction=extraction,
            graph=graph,
            regions=regions,
        )
    
    def _merge_into_lines(
        self,
        blocks: List[TextBlock],
        tolerance: float = 5.0,
    ) -> List[TextBlock]:
        """
        Merge word-level blocks into line-level blocks.
        
        This improves readability and reduces the number of blocks
        while preserving spatial information.
        """
        if not blocks:
            return []
        
        # Group by line (similar y-coordinate)
        lines = group_blocks_by_line(blocks, tolerance)
        
        merged_blocks = []
        
        for line in lines:
            if not line:
                continue
            
            # Sort by x position
            line.sort(key=lambda b: b.bbox.x0)
            
            # Merge consecutive blocks that are close together
            current_group = [line[0]]
            
            for block in line[1:]:
                last_block = current_group[-1]
                gap = block.bbox.x0 - last_block.bbox.x1
                
                # If gap is small (within a few characters), merge
                avg_char_width = last_block.bbox.width / max(len(last_block.text), 1)
                
                if gap < avg_char_width * 3:  # Gap less than 3 character widths
                    current_group.append(block)
                else:
                    # Create merged block for current group
                    merged_blocks.append(self._merge_block_group(current_group))
                    current_group = [block]
            
            # Don't forget last group
            if current_group:
                merged_blocks.append(self._merge_block_group(current_group))
        
        return merged_blocks
    
    def _merge_block_group(self, blocks: List[TextBlock]) -> TextBlock:
        """Merge a group of blocks into one."""
        if len(blocks) == 1:
            return blocks[0]
        
        # Combine text with spaces
        text = ' '.join(b.text for b in blocks)
        
        # Union of bounding boxes
        merged_bbox = blocks[0].bbox
        for block in blocks[1:]:
            merged_bbox = merged_bbox.union(block.bbox)
        
        # Take attributes from first block
        first = blocks[0]
        
        return TextBlock(
            text=text,
            bbox=merged_bbox,
            block_type=first.block_type,
            confidence=min(b.confidence for b in blocks),
            font_name=first.font_name,
            font_size=first.font_size,
            is_bold=first.is_bold,
            is_italic=first.is_italic,
            page_number=first.page_number,
            source=first.source,
        )
