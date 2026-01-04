"""
Table extraction from PDF pages.
Flattens tables into pipe-separated row strings.
No business logic - pure content extraction.
"""

from dataclasses import dataclass
from typing import List, Optional
import pdfplumber


@dataclass
class TableBlock:
    """A table extracted from a PDF page."""
    content: str  # Pipe-separated rows, newline between rows
    block_type: str = 'table'
    top: float = 0
    bottom: float = 0
    page_num: int = 0
    row_count: int = 0
    col_count: int = 0


def extract_tables(page: pdfplumber.page.Page, page_num: int) -> List[TableBlock]:
    """
    Extract all tables from a PDF page.
    
    Args:
        page: pdfplumber page object
        page_num: 1-based page number
        
    Returns:
        List of TableBlock objects
    """
    tables = []
    
    try:
        # Find tables on the page
        page_tables = page.find_tables()
        
        for table_obj in page_tables:
            # Extract table data
            table_data = table_obj.extract()
            
            if not table_data:
                continue
            
            # Get table bounding box
            bbox = table_obj.bbox
            top = bbox[1] if bbox else 0
            bottom = bbox[3] if bbox else 0
            
            # Convert to pipe-separated format
            rows = []
            for row in table_data:
                # Clean cells: replace None with empty string, strip whitespace
                cleaned_cells = []
                for cell in row:
                    if cell is None:
                        cleaned_cells.append('')
                    else:
                        # Clean whitespace and newlines within cells
                        cleaned = str(cell).replace('\n', ' ').strip()
                        cleaned_cells.append(cleaned)
                
                row_str = ' | '.join(cleaned_cells)
                rows.append(row_str)
            
            content = '\n'.join(rows)
            
            tables.append(TableBlock(
                content=content,
                block_type='table',
                top=top,
                bottom=bottom,
                page_num=page_num,
                row_count=len(table_data),
                col_count=len(table_data[0]) if table_data else 0
            ))
            
    except Exception as e:
        # Never fail - log and continue
        print(f"Warning: Table extraction error on page {page_num}: {e}")
    
    return tables


def get_table_regions(page: pdfplumber.page.Page) -> List[tuple]:
    """
    Get bounding boxes of all tables on a page.
    Used to exclude table regions from text extraction.
    
    Returns:
        List of (x0, top, x1, bottom) tuples
    """
    regions = []
    
    try:
        page_tables = page.find_tables()
        for table_obj in page_tables:
            if table_obj.bbox:
                regions.append(table_obj.bbox)
    except Exception:
        pass
    
    return regions
