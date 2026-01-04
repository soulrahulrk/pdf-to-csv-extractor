"""
Continuation Detection

Detects when tables continue across page boundaries.
Uses multiple signals to determine continuation probability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ContinuationConfidence(Enum):
    """Confidence level for continuation detection."""
    
    DEFINITE = auto()    # Strong signals confirm continuation
    LIKELY = auto()      # Multiple signals suggest continuation
    POSSIBLE = auto()    # Some signals, unclear
    UNLIKELY = auto()    # Few signals, probably not
    NONE = auto()        # Definitely not a continuation
    
    @property
    def is_continuation(self) -> bool:
        """Whether this should be treated as continuation."""
        return self in (ContinuationConfidence.DEFINITE, ContinuationConfidence.LIKELY)


@dataclass
class ContinuationSignal:
    """
    A signal that indicates table continuation.
    """
    signal_type: str
    weight: float
    description: str
    evidence: Any = None
    
    @property
    def is_positive(self) -> bool:
        """Whether this signal supports continuation."""
        return self.weight > 0


@dataclass
class ContinuationAnalysis:
    """
    Analysis result for table continuation.
    """
    source_page: int
    target_page: int
    confidence: ContinuationConfidence
    confidence_score: float
    signals: List[ContinuationSignal]
    column_match_score: float
    structural_match_score: float
    textual_signals_score: float
    
    @property
    def positive_signals(self) -> List[ContinuationSignal]:
        """Get positive continuation signals."""
        return [s for s in self.signals if s.is_positive]
    
    @property
    def negative_signals(self) -> List[ContinuationSignal]:
        """Get negative signals."""
        return [s for s in self.signals if not s.is_positive]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_page': self.source_page,
            'target_page': self.target_page,
            'confidence': self.confidence.name,
            'confidence_score': round(self.confidence_score, 3),
            'is_continuation': self.confidence.is_continuation,
            'column_match_score': round(self.column_match_score, 3),
            'structural_match_score': round(self.structural_match_score, 3),
            'textual_signals_score': round(self.textual_signals_score, 3),
            'signals': [
                {'type': s.signal_type, 'weight': s.weight, 'description': s.description}
                for s in self.signals
            ],
        }


class ContinuationDetector:
    """
    Detects table continuation across pages.
    
    Uses multiple signals:
    1. Column alignment - columns match between pages
    2. Header absence - second page has no headers
    3. Row numbering - row numbers continue
    4. Textual signals - "continued", "cont'd", etc.
    5. Structural similarity - similar cell patterns
    6. Vertical position - table at top of next page
    7. Content patterns - data types match
    
    Usage:
        detector = ContinuationDetector()
        
        analysis = detector.analyze(
            table1=first_page_table,
            table2=next_page_table,
            page1_number=1,
            page2_number=2,
        )
        
        if analysis.confidence.is_continuation:
            # Tables should be stitched together
            pass
    """
    
    # Textual continuation indicators
    CONTINUATION_PHRASES = [
        r'\(continued\)',
        r'\(cont\.?\)',
        r'continued from',
        r'continued on',
        r'cont\'d',
        r'continued\s*\.\.\.',
        r'see next page',
        r'see following page',
        r'table continued',
    ]
    
    # Phrases indicating table end
    END_PHRASES = [
        r'total',
        r'subtotal',
        r'grand total',
        r'sum',
        r'end of',
        r'concluded',
    ]
    
    def __init__(
        self,
        column_tolerance: float = 10.0,
        min_column_match: float = 0.7,
        definite_threshold: float = 0.85,
        likely_threshold: float = 0.65,
        possible_threshold: float = 0.45,
    ):
        """
        Initialize continuation detector.
        
        Args:
            column_tolerance: Tolerance for column alignment (points)
            min_column_match: Minimum column match ratio for continuation
            definite_threshold: Score threshold for DEFINITE
            likely_threshold: Score threshold for LIKELY
            possible_threshold: Score threshold for POSSIBLE
        """
        self.column_tolerance = column_tolerance
        self.min_column_match = min_column_match
        self.definite_threshold = definite_threshold
        self.likely_threshold = likely_threshold
        self.possible_threshold = possible_threshold
        
        # Compile regex patterns
        self._continuation_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PHRASES
        ]
        self._end_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.END_PHRASES
        ]
    
    def analyze(
        self,
        table1: Any,
        table2: Any,
        page1_number: int,
        page2_number: int,
    ) -> ContinuationAnalysis:
        """
        Analyze whether table2 is a continuation of table1.
        
        Args:
            table1: First table (TableBoundary or table data)
            table2: Second table (TableBoundary or table data)
            page1_number: Page number of first table
            page2_number: Page number of second table
            
        Returns:
            ContinuationAnalysis with confidence and signals
        """
        signals = []
        
        # Check if pages are consecutive
        if page2_number != page1_number + 1:
            signals.append(ContinuationSignal(
                signal_type='non_consecutive_pages',
                weight=-0.5,
                description=f"Pages not consecutive ({page1_number} -> {page2_number})",
            ))
        
        # Analyze column alignment
        col_score, col_signals = self._analyze_column_alignment(table1, table2)
        signals.extend(col_signals)
        
        # Analyze structural similarity
        struct_score, struct_signals = self._analyze_structure(table1, table2)
        signals.extend(struct_signals)
        
        # Check textual signals
        text_score, text_signals = self._analyze_textual_signals(table1, table2)
        signals.extend(text_signals)
        
        # Check position on page
        pos_signals = self._analyze_position(table2, page2_number)
        signals.extend(pos_signals)
        
        # Check for end indicators on first table
        end_signals = self._check_end_indicators(table1)
        signals.extend(end_signals)
        
        # Calculate final score
        total_weight = sum(s.weight for s in signals)
        
        # Normalize to 0-1 range
        confidence_score = (total_weight + 1) / 2
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Determine confidence level
        if confidence_score >= self.definite_threshold:
            confidence = ContinuationConfidence.DEFINITE
        elif confidence_score >= self.likely_threshold:
            confidence = ContinuationConfidence.LIKELY
        elif confidence_score >= self.possible_threshold:
            confidence = ContinuationConfidence.POSSIBLE
        elif confidence_score >= 0.25:
            confidence = ContinuationConfidence.UNLIKELY
        else:
            confidence = ContinuationConfidence.NONE
        
        return ContinuationAnalysis(
            source_page=page1_number,
            target_page=page2_number,
            confidence=confidence,
            confidence_score=confidence_score,
            signals=signals,
            column_match_score=col_score,
            structural_match_score=struct_score,
            textual_signals_score=text_score,
        )
    
    def _analyze_column_alignment(
        self,
        table1: Any,
        table2: Any,
    ) -> Tuple[float, List[ContinuationSignal]]:
        """Analyze column alignment between tables."""
        signals = []
        
        # Get column positions
        cols1 = self._get_column_positions(table1)
        cols2 = self._get_column_positions(table2)
        
        if not cols1 or not cols2:
            return 0.5, signals
        
        # Check column count
        if len(cols1) == len(cols2):
            signals.append(ContinuationSignal(
                signal_type='same_column_count',
                weight=0.2,
                description=f"Same number of columns ({len(cols1)})",
            ))
        else:
            diff = abs(len(cols1) - len(cols2))
            signals.append(ContinuationSignal(
                signal_type='different_column_count',
                weight=-0.1 * diff,
                description=f"Different column counts ({len(cols1)} vs {len(cols2)})",
            ))
        
        # Check alignment
        matched = 0
        for c1 in cols1:
            for c2 in cols2:
                if abs(c1 - c2) <= self.column_tolerance:
                    matched += 1
                    break
        
        match_ratio = matched / max(len(cols1), len(cols2)) if cols1 else 0
        
        if match_ratio >= 0.9:
            signals.append(ContinuationSignal(
                signal_type='excellent_column_alignment',
                weight=0.3,
                description=f"Excellent column alignment ({match_ratio:.0%})",
            ))
        elif match_ratio >= self.min_column_match:
            signals.append(ContinuationSignal(
                signal_type='good_column_alignment',
                weight=0.2,
                description=f"Good column alignment ({match_ratio:.0%})",
            ))
        elif match_ratio >= 0.5:
            signals.append(ContinuationSignal(
                signal_type='partial_column_alignment',
                weight=0.0,
                description=f"Partial column alignment ({match_ratio:.0%})",
            ))
        else:
            signals.append(ContinuationSignal(
                signal_type='poor_column_alignment',
                weight=-0.3,
                description=f"Poor column alignment ({match_ratio:.0%})",
            ))
        
        return match_ratio, signals
    
    def _analyze_structure(
        self,
        table1: Any,
        table2: Any,
    ) -> Tuple[float, List[ContinuationSignal]]:
        """Analyze structural similarity."""
        signals = []
        score = 0.5
        
        # Check if table2 lacks header
        has_header1 = getattr(table1, 'has_header', True)
        has_header2 = getattr(table2, 'has_header', True)
        
        if has_header1 and not has_header2:
            signals.append(ContinuationSignal(
                signal_type='header_pattern',
                weight=0.25,
                description="First table has header, second doesn't (continuation pattern)",
            ))
            score += 0.2
        elif has_header1 and has_header2:
            signals.append(ContinuationSignal(
                signal_type='both_have_headers',
                weight=-0.2,
                description="Both tables have headers (likely separate tables)",
            ))
            score -= 0.2
        
        # Check width similarity
        width1 = self._get_table_width(table1)
        width2 = self._get_table_width(table2)
        
        if width1 and width2:
            width_ratio = min(width1, width2) / max(width1, width2)
            
            if width_ratio >= 0.95:
                signals.append(ContinuationSignal(
                    signal_type='same_width',
                    weight=0.15,
                    description="Tables have same width",
                ))
                score += 0.15
            elif width_ratio >= 0.8:
                signals.append(ContinuationSignal(
                    signal_type='similar_width',
                    weight=0.05,
                    description=f"Tables have similar width ({width_ratio:.0%})",
                ))
                score += 0.05
        
        return min(1.0, max(0.0, score)), signals
    
    def _analyze_textual_signals(
        self,
        table1: Any,
        table2: Any,
    ) -> Tuple[float, List[ContinuationSignal]]:
        """Check for textual continuation indicators."""
        signals = []
        score = 0.5
        
        # Get text content
        text1 = self._get_table_text(table1)
        text2 = self._get_table_text(table2)
        
        # Check for continuation phrases
        for pattern in self._continuation_patterns:
            if text1 and pattern.search(text1):
                signals.append(ContinuationSignal(
                    signal_type='continuation_phrase_in_source',
                    weight=0.3,
                    description=f"Found continuation indicator in first table",
                    evidence=pattern.pattern,
                ))
                score += 0.25
                break
            
            if text2 and pattern.search(text2):
                signals.append(ContinuationSignal(
                    signal_type='continuation_phrase_in_target',
                    weight=0.25,
                    description=f"Found continuation indicator in second table",
                    evidence=pattern.pattern,
                ))
                score += 0.2
                break
        
        return min(1.0, score), signals
    
    def _analyze_position(
        self,
        table2: Any,
        page_number: int,
    ) -> List[ContinuationSignal]:
        """Analyze table position on page."""
        signals = []
        
        # Check if table is at top of page
        y0 = getattr(table2, 'y0', None)
        
        if y0 is not None:
            # Typical page margin is ~72 points (1 inch)
            if y0 < 100:
                signals.append(ContinuationSignal(
                    signal_type='at_page_top',
                    weight=0.15,
                    description="Table at top of page (common for continuations)",
                ))
            elif y0 > 300:
                signals.append(ContinuationSignal(
                    signal_type='not_at_page_top',
                    weight=-0.1,
                    description="Table not at top of page",
                ))
        
        return signals
    
    def _check_end_indicators(
        self,
        table1: Any,
    ) -> List[ContinuationSignal]:
        """Check if first table has end indicators."""
        signals = []
        
        text = self._get_last_row_text(table1)
        
        if text:
            for pattern in self._end_patterns:
                if pattern.search(text):
                    signals.append(ContinuationSignal(
                        signal_type='end_indicator_found',
                        weight=-0.35,
                        description=f"Found end indicator in first table ('{pattern.pattern}')",
                    ))
                    break
        
        return signals
    
    def _get_column_positions(self, table: Any) -> List[float]:
        """Extract column positions from table."""
        # Handle different table representations
        if hasattr(table, 'column_positions'):
            return table.column_positions
        
        if hasattr(table, 'metadata') and 'column_positions' in table.metadata:
            return table.metadata['column_positions']
        
        # Try to extract from cells
        if hasattr(table, 'cells') and table.cells:
            positions = set()
            for row in table.cells:
                for cell in row:
                    if hasattr(cell, 'x0'):
                        positions.add(cell.x0)
            return sorted(positions)
        
        # Estimate from boundary
        if hasattr(table, 'x0') and hasattr(table, 'x1'):
            col_count = getattr(table, 'column_count_estimate', 3)
            if col_count > 0:
                width = table.x1 - table.x0
                step = width / col_count
                return [table.x0 + i * step for i in range(col_count)]
        
        return []
    
    def _get_table_width(self, table: Any) -> Optional[float]:
        """Get table width."""
        if hasattr(table, 'width'):
            return table.width
        
        if hasattr(table, 'x0') and hasattr(table, 'x1'):
            return table.x1 - table.x0
        
        return None
    
    def _get_table_text(self, table: Any) -> Optional[str]:
        """Get table text content."""
        if hasattr(table, 'text'):
            return table.text
        
        if hasattr(table, 'extract'):
            try:
                return str(table.extract())
            except Exception:
                pass
        
        if hasattr(table, 'cells'):
            try:
                texts = []
                for row in table.cells:
                    for cell in row:
                        if cell:
                            texts.append(str(cell))
                return ' '.join(texts)
            except Exception:
                pass
        
        return None
    
    def _get_last_row_text(self, table: Any) -> Optional[str]:
        """Get text from last row of table."""
        if hasattr(table, 'cells') and table.cells:
            try:
                last_row = table.cells[-1]
                return ' '.join(str(cell) for cell in last_row if cell)
            except Exception:
                pass
        
        return None
    
    def detect_continuation_chain(
        self,
        tables: List[Tuple[int, Any]],
    ) -> List[List[int]]:
        """
        Detect chains of continuing tables across multiple pages.
        
        Args:
            tables: List of (page_number, table) tuples
            
        Returns:
            List of chains, where each chain is a list of page numbers
        """
        if not tables:
            return []
        
        # Sort by page number
        sorted_tables = sorted(tables, key=lambda x: x[0])
        
        chains = []
        current_chain = [sorted_tables[0][0]]
        
        for i in range(len(sorted_tables) - 1):
            page1, table1 = sorted_tables[i]
            page2, table2 = sorted_tables[i + 1]
            
            analysis = self.analyze(table1, table2, page1, page2)
            
            if analysis.confidence.is_continuation:
                current_chain.append(page2)
            else:
                if len(current_chain) > 1:
                    chains.append(current_chain)
                current_chain = [page2]
        
        if len(current_chain) > 1:
            chains.append(current_chain)
        
        return chains
