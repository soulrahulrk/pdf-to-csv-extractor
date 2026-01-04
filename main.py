#!/usr/bin/env python3
"""
Generic PDF Content Extractor

Extracts all readable content from any PDF and outputs to CSV.
No business logic, no field assumptions - pure content extraction.

Usage:
    python main.py -i document.pdf -o content.csv
    python main.py -i ./pdfs/ -o all_content.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber is required. Install with: pip install pdfplumber")
    sys.exit(1)

from extract.text_blocks import extract_text_blocks, get_page_char_count
from extract.tables import extract_tables, get_table_regions
from extract.ocr import extract_ocr_text, should_ocr, is_ocr_available
from output.generic_csv_writer import ContentRow, write_csv, create_empty_row


def process_page(
    page: pdfplumber.page.Page,
    page_num: int,
    source_file: str,
    ocr_threshold: int = 50,
    enable_ocr: bool = True
) -> List[ContentRow]:
    """
    Process a single PDF page and extract all content.
    
    Args:
        page: pdfplumber page object
        page_num: 1-based page number
        source_file: Source filename for CSV
        ocr_threshold: Character count below which OCR is triggered
        enable_ocr: Whether to enable OCR for scanned pages
        
    Returns:
        List of ContentRow objects
    """
    rows = []
    block_index = 0
    
    # Check if page needs OCR
    char_count = get_page_char_count(page)
    use_ocr = enable_ocr and should_ocr(char_count, ocr_threshold)
    
    if use_ocr:
        # Scanned page - use OCR
        ocr_blocks = extract_ocr_text(page, page_num)
        for block in ocr_blocks:
            rows.append(ContentRow(
                source_file=source_file,
                page_number=page_num,
                block_type=block.block_type,
                block_index=block_index,
                content=block.content
            ))
            block_index += 1
    else:
        # Digital page - extract text and tables
        
        # Get table regions to potentially exclude from text
        table_regions = get_table_regions(page)
        
        # Extract tables first
        tables = extract_tables(page, page_num)
        for table in tables:
            rows.append(ContentRow(
                source_file=source_file,
                page_number=page_num,
                block_type=table.block_type,
                block_index=block_index,
                content=table.content
            ))
            block_index += 1
        
        # Extract text blocks
        text_blocks = extract_text_blocks(page, page_num)
        for block in text_blocks:
            # Skip empty blocks if we already have content
            if block.block_type == 'empty' and (tables or any(b.block_type != 'empty' for b in text_blocks)):
                continue
                
            rows.append(ContentRow(
                source_file=source_file,
                page_number=page_num,
                block_type=block.block_type,
                block_index=block_index,
                content=block.content
            ))
            block_index += 1
    
    # If page has no content at all, add empty row
    if not rows:
        rows.append(create_empty_row(source_file, page_num))
    
    return rows


def process_pdf(
    pdf_path: Path,
    ocr_threshold: int = 50,
    enable_ocr: bool = True,
    verbose: bool = False
) -> List[ContentRow]:
    """
    Process a complete PDF file.
    
    Args:
        pdf_path: Path to PDF file
        ocr_threshold: Character count below which OCR is triggered
        enable_ocr: Whether to enable OCR
        verbose: Print progress messages
        
    Returns:
        List of ContentRow objects for all pages
    """
    all_rows = []
    source_file = pdf_path.name
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            if verbose:
                print(f"Processing: {source_file} ({total_pages} pages)")
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                if verbose:
                    print(f"  Page {page_num}/{total_pages}...", end=' ')
                
                try:
                    page_rows = process_page(
                        page, page_num, source_file,
                        ocr_threshold=ocr_threshold,
                        enable_ocr=enable_ocr
                    )
                    all_rows.extend(page_rows)
                    
                    if verbose:
                        print(f"{len(page_rows)} blocks")
                        
                except Exception as e:
                    # Never fail entire document - add error row
                    if verbose:
                        print(f"Error: {e}")
                    all_rows.append(ContentRow(
                        source_file=source_file,
                        page_number=page_num,
                        block_type='empty',
                        block_index=0,
                        content=f'[Extraction error: {str(e)}]'
                    ))
                    
    except Exception as e:
        # PDF open failed - still output something
        if verbose:
            print(f"Failed to open PDF: {e}")
        all_rows.append(ContentRow(
            source_file=source_file,
            page_number=0,
            block_type='empty',
            block_index=0,
            content=f'[Failed to open PDF: {str(e)}]'
        ))
    
    return all_rows


def find_pdfs(input_path: Path) -> List[Path]:
    """Find all PDF files in a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            return [input_path]
        else:
            return []
    elif input_path.is_dir():
        return list(input_path.glob('**/*.pdf'))
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Extract content from PDF files to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i document.pdf -o content.csv
  python main.py -i ./pdfs/ -o all_content.csv
  python main.py -i scan.pdf -o output.csv --ocr-threshold 100

Output CSV columns:
  source_file   - PDF filename
  page_number   - 1-based page number  
  block_type    - paragraph | line | table | ocr_text | empty
  block_index   - 0-based index within page
  content       - Extracted text content
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input PDF file or directory'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--ocr-threshold',
        type=int,
        default=50,
        help='Character count below which OCR is triggered (default: 50)'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR even for scanned pages'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Find PDFs
    pdf_files = find_pdfs(input_path)
    if not pdf_files:
        print(f"Error: No PDF files found in: {args.input}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(pdf_files)} PDF file(s)")
        if not args.no_ocr and is_ocr_available():
            print(f"OCR enabled (threshold: {args.ocr_threshold} chars)")
        elif args.no_ocr:
            print("OCR disabled")
        else:
            print("OCR unavailable (pytesseract not installed)")
    
    # Process all PDFs
    all_rows = []
    for pdf_path in pdf_files:
        rows = process_pdf(
            pdf_path,
            ocr_threshold=args.ocr_threshold,
            enable_ocr=not args.no_ocr,
            verbose=args.verbose
        )
        all_rows.extend(rows)
    
    # Write output
    output_path = Path(args.output)
    rows_written = write_csv(all_rows, output_path)
    
    if args.verbose:
        print(f"\nOutput: {output_path}")
        print(f"Total rows: {rows_written}")
    else:
        print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == '__main__':
    main()
