"""
Text block extraction from PDF pages.
Groups words into logical blocks based on vertical spacing and font characteristics.
No business logic - pure content extraction.
"""

from dataclasses import dataclass
from typing import List, Optional
import pdfplumber


@dataclass
class TextBlock:
    """A block of text content."""
    content: str
    block_type: str  # 'paragraph', 'line', 'empty'
    top: float
    bottom: float
    page_num: int


def extract_text_blocks(page: pdfplumber.page.Page, page_num: int) -> List[TextBlock]:
    """
    Extract text blocks from a PDF page.
    
    Groups words into blocks based on:
    - Vertical spacing between lines
    - Reading order (top to bottom, left to right)
    
    Args:
        page: pdfplumber page object
        page_num: 1-based page number
        
    Returns:
        List of TextBlock objects in reading order
    """
    words = page.extract_words(
        keep_blank_chars=True,
        x_tolerance=3,
        y_tolerance=3,
        extra_attrs=['fontname', 'size']
    )
    
    if not words:
        return [TextBlock(
            content='',
            block_type='empty',
            top=0,
            bottom=0,
            page_num=page_num
        )]
    
    # Sort words by vertical position (top), then horizontal (left)
    words = sorted(words, key=lambda w: (w['top'], w['x0']))
    
    # Group words into lines based on vertical position
    lines = _group_into_lines(words)
    
    # Group lines into blocks based on spacing
    blocks = _group_into_blocks(lines, page_num)
    
    return blocks


def _group_into_lines(words: List[dict]) -> List[List[dict]]:
    """Group words into lines based on vertical position."""
    if not words:
        return []
    
    lines = []
    current_line = [words[0]]
    current_top = words[0]['top']
    
    # Tolerance for same-line detection (words within this vertical distance are same line)
    LINE_TOLERANCE = 5
    
    for word in words[1:]:
        if abs(word['top'] - current_top) <= LINE_TOLERANCE:
            # Same line
            current_line.append(word)
        else:
            # New line - save current and start new
            lines.append(sorted(current_line, key=lambda w: w['x0']))
            current_line = [word]
            current_top = word['top']
    
    # Don't forget last line
    if current_line:
        lines.append(sorted(current_line, key=lambda w: w['x0']))
    
    return lines


def _group_into_blocks(lines: List[List[dict]], page_num: int) -> List[TextBlock]:
    """
    Group lines into blocks based on vertical spacing.
    
    Large gaps between lines indicate paragraph breaks.
    """
    if not lines:
        return [TextBlock(
            content='',
            block_type='empty',
            top=0,
            bottom=0,
            page_num=page_num
        )]
    
    blocks = []
    current_block_lines = []
    current_block_top = None
    prev_bottom = None
    
    # Calculate average line height for spacing threshold
    line_heights = []
    for line in lines:
        if line:
            top = min(w['top'] for w in line)
            bottom = max(w['bottom'] for w in line)
            line_heights.append(bottom - top)
    
    avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 12
    
    # Spacing threshold: if gap > 1.5x line height, new block
    BLOCK_GAP_MULTIPLIER = 1.5
    
    for line in lines:
        if not line:
            continue
            
        line_top = min(w['top'] for w in line)
        line_bottom = max(w['bottom'] for w in line)
        line_text = ' '.join(w['text'] for w in line)
        
        if current_block_top is None:
            # First line
            current_block_top = line_top
            current_block_lines.append(line_text)
            prev_bottom = line_bottom
        else:
            # Check gap from previous line
            gap = line_top - prev_bottom
            
            if gap > avg_line_height * BLOCK_GAP_MULTIPLIER:
                # New block - save current
                block_content = '\n'.join(current_block_lines)
                block_type = 'paragraph' if len(current_block_lines) > 1 else 'line'
                
                blocks.append(TextBlock(
                    content=block_content.strip(),
                    block_type=block_type,
                    top=current_block_top,
                    bottom=prev_bottom,
                    page_num=page_num
                ))
                
                # Start new block
                current_block_lines = [line_text]
                current_block_top = line_top
            else:
                # Continue current block
                current_block_lines.append(line_text)
            
            prev_bottom = line_bottom
    
    # Save final block
    if current_block_lines:
        block_content = '\n'.join(current_block_lines)
        block_type = 'paragraph' if len(current_block_lines) > 1 else 'line'
        
        blocks.append(TextBlock(
            content=block_content.strip(),
            block_type=block_type,
            top=current_block_top,
            bottom=prev_bottom,
            page_num=page_num
        ))
    
    return blocks


def get_page_char_count(page: pdfplumber.page.Page) -> int:
    """Get total character count from a page (for OCR threshold check)."""
    text = page.extract_text() or ''
    return len(text.strip())
