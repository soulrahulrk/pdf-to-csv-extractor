"""
Table Stitching

Merges tables that span multiple pages into unified structures.
Handles header propagation, row alignment, and data consistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Iterator
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class TableCell:
    """
    Represents a single cell in a table.
    """
    value: Any
    row_index: int
    column_index: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    source_page: int = 0
    confidence: float = 1.0
    original_bbox: Optional[Tuple[float, float, float, float]] = None
    
    @property
    def is_merged(self) -> bool:
        """Check if cell spans multiple rows/columns."""
        return self.rowspan > 1 or self.colspan > 1
    
    @property
    def text(self) -> str:
        """Get cell value as text."""
        if self.value is None:
            return ''
        return str(self.value).strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'row_index': self.row_index,
            'column_index': self.column_index,
            'rowspan': self.rowspan,
            'colspan': self.colspan,
            'is_header': self.is_header,
            'source_page': self.source_page,
            'confidence': round(self.confidence, 3),
        }


@dataclass
class TableRow:
    """
    Represents a row in a table.
    """
    cells: List[TableCell]
    row_index: int
    is_header: bool = False
    source_page: int = 0
    original_y: Optional[float] = None
    
    @property
    def column_count(self) -> int:
        """Number of columns (accounting for colspan)."""
        return sum(cell.colspan for cell in self.cells)
    
    @property
    def values(self) -> List[Any]:
        """Get cell values."""
        return [cell.value for cell in self.cells]
    
    @property
    def text_values(self) -> List[str]:
        """Get cell values as text."""
        return [cell.text for cell in self.cells]
    
    def get_cell(self, column_index: int) -> Optional[TableCell]:
        """Get cell by column index."""
        current_col = 0
        for cell in self.cells:
            if current_col <= column_index < current_col + cell.colspan:
                return cell
            current_col += cell.colspan
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'row_index': self.row_index,
            'is_header': self.is_header,
            'source_page': self.source_page,
            'cells': [cell.to_dict() for cell in self.cells],
        }


@dataclass
class StitchedTable:
    """
    A table that may span multiple pages, stitched together.
    """
    table_id: str
    rows: List[TableRow]
    header_rows: List[TableRow]
    column_count: int
    source_pages: List[int]
    column_names: List[str] = field(default_factory=list)
    column_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def row_count(self) -> int:
        """Total data rows (excluding headers)."""
        return len(self.rows)
    
    @property
    def total_rows(self) -> int:
        """Total rows including headers."""
        return len(self.header_rows) + len(self.rows)
    
    @property
    def is_multi_page(self) -> bool:
        """Whether table spans multiple pages."""
        return len(self.source_pages) > 1
    
    @property
    def page_range(self) -> str:
        """Human-readable page range."""
        if not self.source_pages:
            return "unknown"
        if len(self.source_pages) == 1:
            return str(self.source_pages[0])
        return f"{min(self.source_pages)}-{max(self.source_pages)}"
    
    def get_row(self, index: int) -> Optional[TableRow]:
        """Get data row by index."""
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None
    
    def get_column(self, index: int) -> List[Any]:
        """Get all values in a column."""
        values = []
        for row in self.rows:
            cell = row.get_cell(index)
            values.append(cell.value if cell else None)
        return values
    
    def get_column_by_name(self, name: str) -> List[Any]:
        """Get column values by header name."""
        try:
            index = self.column_names.index(name)
            return self.get_column(index)
        except ValueError:
            # Try case-insensitive match
            for i, col_name in enumerate(self.column_names):
                if col_name.lower() == name.lower():
                    return self.get_column(i)
        return []
    
    def iter_rows(self, include_header: bool = False) -> Iterator[TableRow]:
        """Iterate over rows."""
        if include_header:
            yield from self.header_rows
        yield from self.rows
    
    def to_list(self, include_header: bool = True) -> List[List[Any]]:
        """Convert to list of lists."""
        result = []
        
        if include_header:
            for row in self.header_rows:
                result.append(row.values)
        
        for row in self.rows:
            result.append(row.values)
        
        return result
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries (using headers as keys)."""
        if not self.column_names:
            return []
        
        result = []
        for row in self.rows:
            row_dict = {}
            for i, name in enumerate(self.column_names):
                cell = row.get_cell(i)
                row_dict[name] = cell.value if cell else None
            result.append(row_dict)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'table_id': self.table_id,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'source_pages': self.source_pages,
            'is_multi_page': self.is_multi_page,
            'column_names': self.column_names,
            'column_types': self.column_types,
            'header_rows': [r.to_dict() for r in self.header_rows],
            'data_rows': [r.to_dict() for r in self.rows],
        }


@dataclass
class StitchingConfig:
    """
    Configuration for table stitching.
    """
    # Column alignment tolerance
    column_tolerance: float = 10.0
    
    # Minimum confidence for continuation
    min_continuation_confidence: float = 0.6
    
    # Whether to propagate headers to continued tables
    propagate_headers: bool = True
    
    # Whether to validate column alignment
    validate_columns: bool = True
    
    # Maximum column count mismatch allowed
    max_column_mismatch: int = 1
    
    # Whether to attempt column type inference
    infer_column_types: bool = True
    
    # Whether to normalize column widths
    normalize_columns: bool = True


class TableStitcher:
    """
    Stitches tables across page boundaries.
    
    Features:
    - Header detection and propagation
    - Column alignment validation
    - Row continuation handling
    - Column type inference
    - Multi-page chain stitching
    
    Usage:
        stitcher = TableStitcher()
        
        # Stitch two tables
        stitched = stitcher.stitch(table1, table2)
        
        # Stitch multiple tables
        stitched = stitcher.stitch_chain([table1, table2, table3])
    """
    
    def __init__(self, config: Optional[StitchingConfig] = None):
        """
        Initialize table stitcher.
        
        Args:
            config: Stitching configuration
        """
        self.config = config or StitchingConfig()
    
    def stitch(
        self,
        table1: Any,
        table2: Any,
        table1_page: int = 0,
        table2_page: int = 1,
    ) -> StitchedTable:
        """
        Stitch two tables together.
        
        Args:
            table1: First table (with headers)
            table2: Continuation table
            table1_page: Page number of first table
            table2_page: Page number of second table
            
        Returns:
            StitchedTable combining both tables
        """
        # Convert to internal representation
        rows1 = self._extract_rows(table1, table1_page)
        rows2 = self._extract_rows(table2, table2_page)
        
        if not rows1:
            logger.warning("First table has no rows")
            return self._create_empty_table([table1_page, table2_page])
        
        # Detect headers in first table
        header_rows, data_rows1 = self._split_headers(rows1)
        
        # Validate column alignment if enabled
        if self.config.validate_columns and rows2:
            if not self._validate_column_alignment(rows1, rows2):
                logger.warning("Column alignment mismatch between tables")
        
        # Handle second table (skip repeated headers)
        data_rows2 = self._skip_repeated_headers(rows2, header_rows)
        
        # Renumber rows
        all_data_rows = self._renumber_rows(data_rows1 + data_rows2)
        
        # Extract column names from headers
        column_names = self._extract_column_names(header_rows)
        column_count = max(len(column_names), max((r.column_count for r in all_data_rows), default=0))
        
        # Infer column types
        column_types = []
        if self.config.infer_column_types:
            column_types = self._infer_column_types(all_data_rows, column_count)
        
        return StitchedTable(
            table_id=f"table_{table1_page}_{table2_page}",
            rows=all_data_rows,
            header_rows=header_rows,
            column_count=column_count,
            source_pages=[table1_page, table2_page],
            column_names=column_names,
            column_types=column_types,
        )
    
    def stitch_chain(
        self,
        tables: List[Tuple[int, Any]],
    ) -> StitchedTable:
        """
        Stitch a chain of tables from multiple pages.
        
        Args:
            tables: List of (page_number, table) tuples
            
        Returns:
            Single StitchedTable combining all tables
        """
        if not tables:
            return self._create_empty_table([])
        
        if len(tables) == 1:
            page, table = tables[0]
            rows = self._extract_rows(table, page)
            header_rows, data_rows = self._split_headers(rows)
            column_names = self._extract_column_names(header_rows)
            
            return StitchedTable(
                table_id=f"table_{page}",
                rows=data_rows,
                header_rows=header_rows,
                column_count=max(len(column_names), max((r.column_count for r in data_rows), default=0)),
                source_pages=[page],
                column_names=column_names,
            )
        
        # Sort by page number
        sorted_tables = sorted(tables, key=lambda x: x[0])
        
        # Start with first table
        first_page, first_table = sorted_tables[0]
        result = self.stitch(
            first_table,
            sorted_tables[1][1],
            first_page,
            sorted_tables[1][0],
        )
        
        # Add remaining tables
        for page, table in sorted_tables[2:]:
            rows = self._extract_rows(table, page)
            data_rows = self._skip_repeated_headers(rows, result.header_rows)
            
            # Add to result
            start_idx = len(result.rows)
            for i, row in enumerate(data_rows):
                row.row_index = start_idx + i
            
            result.rows.extend(data_rows)
            result.source_pages.append(page)
        
        # Update table ID
        result.table_id = f"table_{'_'.join(map(str, result.source_pages))}"
        
        return result
    
    def _extract_rows(
        self,
        table: Any,
        page_number: int,
    ) -> List[TableRow]:
        """Extract rows from various table representations."""
        rows = []
        
        # Handle list of lists
        if isinstance(table, list):
            for i, row_data in enumerate(table):
                if isinstance(row_data, list):
                    cells = [
                        TableCell(
                            value=v,
                            row_index=i,
                            column_index=j,
                            source_page=page_number,
                        )
                        for j, v in enumerate(row_data)
                    ]
                    rows.append(TableRow(
                        cells=cells,
                        row_index=i,
                        source_page=page_number,
                    ))
            return rows
        
        # Handle pdfplumber table
        if hasattr(table, 'extract'):
            try:
                data = table.extract()
                for i, row_data in enumerate(data or []):
                    cells = [
                        TableCell(
                            value=v,
                            row_index=i,
                            column_index=j,
                            source_page=page_number,
                        )
                        for j, v in enumerate(row_data or [])
                    ]
                    rows.append(TableRow(
                        cells=cells,
                        row_index=i,
                        source_page=page_number,
                    ))
                return rows
            except Exception as e:
                logger.warning(f"Failed to extract table: {e}")
        
        # Handle cells attribute
        if hasattr(table, 'cells'):
            for i, row_cells in enumerate(table.cells):
                cells = [
                    TableCell(
                        value=cell.value if hasattr(cell, 'value') else cell,
                        row_index=i,
                        column_index=j,
                        source_page=page_number,
                    )
                    for j, cell in enumerate(row_cells)
                ]
                rows.append(TableRow(
                    cells=cells,
                    row_index=i,
                    source_page=page_number,
                ))
            return rows
        
        # Handle rows attribute
        if hasattr(table, 'rows'):
            for i, row in enumerate(table.rows):
                if isinstance(row, TableRow):
                    row.source_page = page_number
                    row.row_index = i
                    rows.append(row)
                elif isinstance(row, list):
                    cells = [
                        TableCell(
                            value=v,
                            row_index=i,
                            column_index=j,
                            source_page=page_number,
                        )
                        for j, v in enumerate(row)
                    ]
                    rows.append(TableRow(
                        cells=cells,
                        row_index=i,
                        source_page=page_number,
                    ))
        
        return rows
    
    def _split_headers(
        self,
        rows: List[TableRow],
    ) -> Tuple[List[TableRow], List[TableRow]]:
        """Split rows into headers and data."""
        if not rows:
            return [], []
        
        header_rows = []
        data_rows = []
        
        # Heuristics for header detection
        for i, row in enumerate(rows):
            is_header = False
            
            # First row is often header
            if i == 0:
                is_header = self._row_looks_like_header(row)
            
            # Check for header markers
            if row.is_header:
                is_header = True
            
            if is_header:
                row.is_header = True
                for cell in row.cells:
                    cell.is_header = True
                header_rows.append(row)
            else:
                data_rows.append(row)
        
        # If no headers detected, assume first row is header
        if not header_rows and rows:
            first_row = rows[0]
            first_row.is_header = True
            for cell in first_row.cells:
                cell.is_header = True
            header_rows.append(first_row)
            data_rows = rows[1:]
        
        return header_rows, data_rows
    
    def _row_looks_like_header(self, row: TableRow) -> bool:
        """Determine if row looks like a header."""
        if not row.cells:
            return False
        
        numeric_count = 0
        empty_count = 0
        
        for cell in row.cells:
            text = cell.text
            
            if not text:
                empty_count += 1
                continue
            
            # Headers rarely contain mostly numbers
            if self._is_numeric(text):
                numeric_count += 1
        
        total = len(row.cells)
        
        # Headers usually have few numeric values
        if numeric_count > total * 0.5:
            return False
        
        # Headers usually aren't mostly empty
        if empty_count > total * 0.7:
            return False
        
        return True
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is primarily numeric."""
        if not text:
            return False
        
        # Remove common formatting
        cleaned = text.replace(',', '').replace('$', '').replace('%', '')
        cleaned = cleaned.replace('(', '').replace(')', '').strip()
        
        try:
            float(cleaned)
            return True
        except ValueError:
            pass
        
        # Check if mostly digits
        digits = sum(1 for c in text if c.isdigit())
        return digits > len(text) * 0.5
    
    def _skip_repeated_headers(
        self,
        rows: List[TableRow],
        header_rows: List[TableRow],
    ) -> List[TableRow]:
        """Skip rows that repeat the header."""
        if not rows or not header_rows:
            return rows
        
        # Get header values for comparison
        header_values = []
        for row in header_rows:
            header_values.append([c.text.lower() for c in row.cells])
        
        # Find where data starts
        data_start = 0
        for i, row in enumerate(rows):
            row_values = [c.text.lower() for c in row.cells]
            
            # Check if this row matches any header row
            is_header = False
            for hv in header_values:
                if self._rows_match(row_values, hv):
                    is_header = True
                    break
            
            if not is_header:
                data_start = i
                break
            
            data_start = i + 1
        
        return rows[data_start:]
    
    def _rows_match(
        self,
        row1: List[str],
        row2: List[str],
        threshold: float = 0.8,
    ) -> bool:
        """Check if two rows match."""
        if len(row1) != len(row2):
            return False
        
        if not row1:
            return True
        
        matches = sum(1 for a, b in zip(row1, row2) if a == b)
        return matches / len(row1) >= threshold
    
    def _validate_column_alignment(
        self,
        rows1: List[TableRow],
        rows2: List[TableRow],
    ) -> bool:
        """Validate that column counts are compatible."""
        if not rows1 or not rows2:
            return True
        
        col_count1 = max(r.column_count for r in rows1)
        col_count2 = max(r.column_count for r in rows2)
        
        return abs(col_count1 - col_count2) <= self.config.max_column_mismatch
    
    def _renumber_rows(self, rows: List[TableRow]) -> List[TableRow]:
        """Renumber rows sequentially."""
        for i, row in enumerate(rows):
            row.row_index = i
            for cell in row.cells:
                cell.row_index = i
        return rows
    
    def _extract_column_names(
        self,
        header_rows: List[TableRow],
    ) -> List[str]:
        """Extract column names from header rows."""
        if not header_rows:
            return []
        
        # Use first header row for names
        first_header = header_rows[0]
        
        names = []
        for i, cell in enumerate(first_header.cells):
            name = cell.text
            
            # Handle empty headers
            if not name:
                name = f"Column_{i + 1}"
            
            # Handle duplicate names
            base_name = name
            counter = 1
            while name in names:
                counter += 1
                name = f"{base_name}_{counter}"
            
            names.append(name)
        
        return names
    
    def _infer_column_types(
        self,
        rows: List[TableRow],
        column_count: int,
    ) -> List[str]:
        """Infer column types from data."""
        if not rows:
            return ['text'] * column_count
        
        types = []
        
        for col_idx in range(column_count):
            # Collect values for this column
            values = []
            for row in rows[:50]:  # Sample first 50 rows
                cell = row.get_cell(col_idx)
                if cell and cell.text:
                    values.append(cell.text)
            
            # Infer type
            col_type = self._infer_type(values)
            types.append(col_type)
        
        return types
    
    def _infer_type(self, values: List[str]) -> str:
        """Infer type from sample values."""
        if not values:
            return 'text'
        
        type_counts = {
            'integer': 0,
            'decimal': 0,
            'currency': 0,
            'percentage': 0,
            'date': 0,
            'text': 0,
        }
        
        for value in values:
            detected = self._detect_value_type(value)
            type_counts[detected] += 1
        
        # Return most common type
        return max(type_counts, key=type_counts.get)
    
    def _detect_value_type(self, value: str) -> str:
        """Detect type of a single value."""
        if not value:
            return 'text'
        
        # Currency
        if value.startswith('$') or value.startswith('£') or value.startswith('€'):
            return 'currency'
        
        # Percentage
        if value.endswith('%'):
            return 'percentage'
        
        # Clean for numeric check
        cleaned = value.replace(',', '').replace('$', '').replace('%', '')
        cleaned = cleaned.replace('(', '-').replace(')', '').strip()
        
        try:
            num = float(cleaned)
            if '.' in cleaned:
                return 'decimal'
            return 'integer'
        except ValueError:
            pass
        
        # Date patterns
        import re
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{1,2}-\d{1,2}-\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return 'date'
        
        return 'text'
    
    def _create_empty_table(self, pages: List[int]) -> StitchedTable:
        """Create an empty stitched table."""
        return StitchedTable(
            table_id='empty',
            rows=[],
            header_rows=[],
            column_count=0,
            source_pages=pages,
        )


def stitch_tables_from_pdf(
    pdf_path: str,
    config: Optional[StitchingConfig] = None,
) -> List[StitchedTable]:
    """
    Convenience function to extract and stitch tables from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        config: Stitching configuration
        
    Returns:
        List of stitched tables
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber required for PDF table extraction")
        return []
    
    from .table_detector import TableDetector
    from .continuation_detector import ContinuationDetector
    
    stitcher = TableStitcher(config)
    detector = TableDetector()
    continuation = ContinuationDetector()
    
    # Extract all tables
    all_tables: List[Tuple[int, Any]] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = detector.detect_from_pdfplumber(page, page_num)
            
            for boundary in tables:
                # Find corresponding pdfplumber table
                plumber_tables = page.find_tables()
                for pt in plumber_tables:
                    if _boundaries_match(boundary, pt):
                        all_tables.append((page_num, pt))
                        break
    
    if not all_tables:
        return []
    
    # Detect continuation chains
    chains = continuation.detect_continuation_chain(all_tables)
    
    # Stitch chains
    stitched = []
    used_pages = set()
    
    for chain in chains:
        chain_tables = [t for t in all_tables if t[0] in chain]
        if chain_tables:
            stitched_table = stitcher.stitch_chain(chain_tables)
            stitched.append(stitched_table)
            used_pages.update(chain)
    
    # Add single-page tables
    for page_num, table in all_tables:
        if page_num not in used_pages:
            rows = stitcher._extract_rows(table, page_num)
            header_rows, data_rows = stitcher._split_headers(rows)
            column_names = stitcher._extract_column_names(header_rows)
            
            stitched_table = StitchedTable(
                table_id=f"table_{page_num}",
                rows=data_rows,
                header_rows=header_rows,
                column_count=max(len(column_names), max((r.column_count for r in data_rows), default=0)),
                source_pages=[page_num],
                column_names=column_names,
            )
            stitched.append(stitched_table)
    
    return stitched


def _boundaries_match(boundary: Any, plumber_table: Any) -> bool:
    """Check if boundary matches pdfplumber table."""
    try:
        bbox = plumber_table.bbox
        return (
            abs(boundary.x0 - bbox[0]) < 5 and
            abs(boundary.y0 - bbox[1]) < 5 and
            abs(boundary.x1 - bbox[2]) < 5 and
            abs(boundary.y1 - bbox[3]) < 5
        )
    except Exception:
        return False
