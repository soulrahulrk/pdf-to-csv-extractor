"""
Table Extraction Module

This module handles extraction of tabular data from PDFs using specialized
table detection libraries. Tables are notoriously difficult to extract from
PDFs because the visual layout often doesn't match the underlying structure.

Strategy:
1. Use Camelot (preferred for bordered tables)
2. Fall back to Tabula (Java-based, more robust for some layouts)
3. Fall back to manual extraction using text positioning

Why multiple libraries?
- Camelot excels at tables with visible borders/lines
- Tabula handles tables without borders better
- Neither is perfect; we use both and pick the best result

Known limitations:
- Merged cells often break extraction
- Tables spanning multiple pages are not automatically merged
- Very complex nested tables may not extract correctly
- OCR quality affects table detection on scanned documents
"""

from pathlib import Path
from typing import Optional, Union
import re

import pandas as pd
from loguru import logger

from .utils import (
    ExtractionResult,
    ExtractionMethod,
    normalize_text,
)


class TableExtractor:
    """
    Extracts tables from PDF documents.
    
    Usage:
        extractor = TableExtractor()
        tables = extractor.extract_tables(Path("invoice.pdf"))
        for df in tables:
            print(df)
    """
    
    def __init__(self, flavor: str = "auto"):
        """
        Initialize table extractor.
        
        Args:
            flavor: Extraction method preference
                   "auto" - Try camelot lattice, then stream, then tabula
                   "lattice" - Use camelot lattice (for bordered tables)
                   "stream" - Use camelot stream (for borderless tables)
                   "tabula" - Use tabula-py
        """
        self.flavor = flavor
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which table extraction libraries are available."""
        self.has_camelot = False
        self.has_tabula = False
        
        # Check Camelot
        try:
            import camelot
            self.camelot = camelot
            self.has_camelot = True
            logger.debug("Camelot available for table extraction")
        except ImportError:
            logger.warning(
                "Camelot not installed. Install with: pip install camelot-py[cv]"
            )
        
        # Check Tabula
        try:
            import tabula
            self.tabula = tabula
            self.has_tabula = True
            logger.debug("Tabula available for table extraction")
        except ImportError:
            logger.warning(
                "Tabula not installed. Install with: pip install tabula-py\n"
                "Note: Tabula requires Java to be installed"
            )
        
        if not self.has_camelot and not self.has_tabula:
            logger.error(
                "No table extraction library available. "
                "Install camelot-py or tabula-py for table extraction."
            )
    
    def extract_tables(
        self,
        pdf_path: Path,
        pages: Optional[str] = "all",
        table_areas: Optional[list[str]] = None
    ) -> list[pd.DataFrame]:
        """
        Extract all tables from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            pages: Page specification ("all", "1", "1,3,5", "1-5")
            table_areas: Optional list of bounding boxes ["x1,y1,x2,y2"]
            
        Returns:
            List of pandas DataFrames, one per detected table
        """
        logger.info(f"Extracting tables from: {pdf_path}")
        
        if self.flavor == "auto":
            return self._extract_auto(pdf_path, pages, table_areas)
        elif self.flavor == "lattice" and self.has_camelot:
            return self._extract_camelot(pdf_path, pages, "lattice", table_areas)
        elif self.flavor == "stream" and self.has_camelot:
            return self._extract_camelot(pdf_path, pages, "stream", table_areas)
        elif self.flavor == "tabula" and self.has_tabula:
            return self._extract_tabula(pdf_path, pages, table_areas)
        else:
            logger.error(f"Requested flavor '{self.flavor}' not available")
            return []
    
    def _extract_auto(
        self,
        pdf_path: Path,
        pages: str,
        table_areas: Optional[list[str]]
    ) -> list[pd.DataFrame]:
        """
        Automatic extraction strategy: try multiple methods and return best results.
        """
        all_tables = []
        
        # Strategy 1: Camelot lattice (best for bordered tables)
        if self.has_camelot:
            try:
                lattice_tables = self._extract_camelot(
                    pdf_path, pages, "lattice", table_areas
                )
                if lattice_tables:
                    logger.debug(f"Camelot lattice found {len(lattice_tables)} tables")
                    all_tables.extend(lattice_tables)
            except Exception as e:
                logger.debug(f"Camelot lattice failed: {e}")
        
        # Strategy 2: Camelot stream (for borderless tables)
        # Only try if lattice didn't find anything
        if not all_tables and self.has_camelot:
            try:
                stream_tables = self._extract_camelot(
                    pdf_path, pages, "stream", table_areas
                )
                if stream_tables:
                    logger.debug(f"Camelot stream found {len(stream_tables)} tables")
                    all_tables.extend(stream_tables)
            except Exception as e:
                logger.debug(f"Camelot stream failed: {e}")
        
        # Strategy 3: Tabula as fallback
        if not all_tables and self.has_tabula:
            try:
                tabula_tables = self._extract_tabula(pdf_path, pages, table_areas)
                if tabula_tables:
                    logger.debug(f"Tabula found {len(tabula_tables)} tables")
                    all_tables.extend(tabula_tables)
            except Exception as e:
                logger.debug(f"Tabula failed: {e}")
        
        # Clean and deduplicate tables
        cleaned_tables = [self._clean_table(t) for t in all_tables]
        cleaned_tables = [t for t in cleaned_tables if not t.empty]
        
        logger.info(f"Total tables extracted: {len(cleaned_tables)}")
        return cleaned_tables
    
    def _extract_camelot(
        self,
        pdf_path: Path,
        pages: str,
        flavor: str,
        table_areas: Optional[list[str]]
    ) -> list[pd.DataFrame]:
        """Extract tables using Camelot."""
        tables = []
        
        try:
            kwargs = {
                'pages': pages,
                'flavor': flavor,
            }
            
            # Flavor-specific settings
            if flavor == 'stream':
                kwargs['edge_tol'] = 50  # More tolerant edge detection
                kwargs['row_tol'] = 10
            
            if table_areas:
                kwargs['table_areas'] = table_areas
            
            table_list = self.camelot.read_pdf(str(pdf_path), **kwargs)
            
            for table in table_list:
                df = table.df
                
                # Camelot provides accuracy metrics
                accuracy = table.accuracy
                logger.debug(f"Table accuracy: {accuracy:.1f}%")
                
                # Skip very low accuracy tables (likely false positives)
                if accuracy < 50:
                    logger.debug("Skipping low-accuracy table")
                    continue
                
                tables.append(df)
                
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            raise
        
        return tables
    
    def _extract_tabula(
        self,
        pdf_path: Path,
        pages: str,
        table_areas: Optional[list[str]]
    ) -> list[pd.DataFrame]:
        """Extract tables using Tabula."""
        tables = []
        
        try:
            # Convert page spec to Tabula format
            tabula_pages = pages if pages != "all" else "all"
            
            kwargs = {
                'pages': tabula_pages,
                'multiple_tables': True,
                'pandas_options': {'header': None},  # Don't assume first row is header
            }
            
            if table_areas:
                # Tabula uses different coordinate system (top-left origin)
                kwargs['area'] = table_areas
            
            table_list = self.tabula.read_pdf(str(pdf_path), **kwargs)
            
            for df in table_list:
                if df is not None and not df.empty:
                    tables.append(df)
                    
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
            raise
        
        return tables
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean extracted table data.
        
        Steps:
        1. Remove completely empty rows/columns
        2. Normalize whitespace in cell values
        3. Attempt to detect and set proper headers
        4. Convert numeric columns
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Step 1: Clean cell values
        for col in df.columns:
            if df[col].dtype == object:
                # Normalize whitespace
                df[col] = df[col].astype(str).apply(
                    lambda x: ' '.join(x.split()) if pd.notna(x) else ''
                )
                # Replace 'nan' strings with empty
                df[col] = df[col].replace('nan', '')
        
        # Step 2: Remove empty rows
        df = df.replace('', pd.NA)
        df = df.dropna(how='all')
        
        # Step 3: Remove empty columns
        df = df.dropna(axis=1, how='all')
        
        if df.empty:
            return df
        
        # Step 4: Try to detect headers
        # Heuristic: first row is header if it looks different from data rows
        first_row = df.iloc[0]
        is_likely_header = self._looks_like_header(first_row, df)
        
        if is_likely_header:
            # Use first row as column names
            new_headers = [str(h).strip() if pd.notna(h) else f'col_{i}' 
                         for i, h in enumerate(first_row)]
            # Make headers unique
            new_headers = self._make_unique_headers(new_headers)
            df.columns = new_headers
            df = df.iloc[1:].reset_index(drop=True)
        else:
            # Use generic column names
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
        
        # Step 5: Try to convert numeric columns
        df = self._convert_numeric_columns(df)
        
        # Reset NA back to empty strings for consistency
        df = df.fillna('')
        
        return df
    
    def _looks_like_header(self, first_row: pd.Series, df: pd.DataFrame) -> bool:
        """
        Heuristic to determine if first row is a header.
        
        Headers typically:
        - Are shorter than data values
        - Contain words like "Description", "Amount", etc.
        - Don't contain numbers (in invoice contexts)
        """
        header_keywords = [
            'description', 'item', 'quantity', 'qty', 'price', 'amount',
            'total', 'rate', 'unit', 'date', 'number', 'no', 'product',
            'service', 'tax', 'discount', 'subtotal', 'name', 'detail'
        ]
        
        # Check if any cell contains header keywords
        for val in first_row:
            if pd.isna(val):
                continue
            val_lower = str(val).lower()
            for keyword in header_keywords:
                if keyword in val_lower:
                    return True
        
        # Check if first row is less numeric than other rows
        first_row_numeric = sum(
            1 for v in first_row 
            if pd.notna(v) and self._is_numeric(str(v))
        )
        
        if len(df) > 1:
            other_row_numeric = sum(
                1 for v in df.iloc[1] 
                if pd.notna(v) and self._is_numeric(str(v))
            )
            if other_row_numeric > first_row_numeric:
                return True
        
        return False
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string value represents a number."""
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£¥₹,\s]', '', value)
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def _make_unique_headers(self, headers: list[str]) -> list[str]:
        """Make column headers unique by appending numbers to duplicates."""
        seen = {}
        result = []
        
        for header in headers:
            if header in seen:
                seen[header] += 1
                result.append(f"{header}_{seen[header]}")
            else:
                seen[header] = 0
                result.append(header)
        
        return result
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to convert columns that appear numeric."""
        for col in df.columns:
            # Skip if column is empty
            non_empty = df[col][df[col].notna() & (df[col] != '')]
            if len(non_empty) == 0:
                continue
            
            # Check what percentage of values are numeric
            numeric_count = sum(1 for v in non_empty if self._is_numeric(str(v)))
            numeric_ratio = numeric_count / len(non_empty)
            
            if numeric_ratio > 0.7:  # If >70% numeric, convert the column
                df[col] = df[col].apply(self._to_numeric)
        
        return df
    
    def _to_numeric(self, value):
        """Convert a value to numeric, handling currency symbols."""
        if pd.isna(value) or value == '':
            return value
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£¥₹,\s]', '', str(value))
        
        try:
            if '.' in cleaned:
                return float(cleaned)
            else:
                return int(cleaned)
        except ValueError:
            return value
    
    def extract_line_items(
        self,
        pdf_path: Path,
        column_mappings: dict[str, list[str]],
        pages: str = "all"
    ) -> list[dict]:
        """
        Extract line items from invoice tables.
        
        This method specifically handles invoice line item tables,
        mapping detected columns to standard field names.
        
        Args:
            pdf_path: Path to PDF
            column_mappings: Dict mapping standard names to possible header variations
                            e.g., {"description": ["Description", "Item", "Product"]}
            pages: Page specification
            
        Returns:
            List of dicts, each representing a line item
        """
        tables = self.extract_tables(pdf_path, pages)
        line_items = []
        
        for df in tables:
            # Try to identify if this is a line items table
            if not self._is_line_item_table(df, column_mappings):
                continue
            
            # Map columns to standard names
            column_map = self._map_columns(df.columns, column_mappings)
            
            if not column_map:
                continue
            
            # Extract rows as dicts
            for _, row in df.iterrows():
                item = {}
                for std_name, actual_col in column_map.items():
                    value = row.get(actual_col, '')
                    if pd.notna(value) and str(value).strip():
                        item[std_name] = str(value).strip()
                
                # Only add if has meaningful content
                if item and any(v for v in item.values()):
                    line_items.append(item)
        
        return line_items
    
    def _is_line_item_table(
        self,
        df: pd.DataFrame,
        column_mappings: dict[str, list[str]]
    ) -> bool:
        """Check if a table appears to be a line items table."""
        if df.empty or len(df) < 2:  # Need at least header + 1 row
            return False
        
        # Get all possible header keywords
        all_keywords = []
        for variations in column_mappings.values():
            all_keywords.extend([v.lower() for v in variations])
        
        # Check if column names match expected patterns
        matched = 0
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in all_keywords:
                if keyword in col_lower:
                    matched += 1
                    break
        
        # Need at least 2 matching columns to consider it a line item table
        return matched >= 2
    
    def _map_columns(
        self,
        columns: list[str],
        column_mappings: dict[str, list[str]]
    ) -> dict[str, str]:
        """Map actual column names to standard field names."""
        mapping = {}
        
        for std_name, variations in column_mappings.items():
            for col in columns:
                col_lower = str(col).lower()
                for variation in variations:
                    if variation.lower() in col_lower:
                        mapping[std_name] = col
                        break
                if std_name in mapping:
                    break
        
        return mapping
    
    def tables_to_text(self, tables: list[pd.DataFrame]) -> str:
        """
        Convert extracted tables to text format.
        
        Useful when table structure isn't important and we just
        need the content for text-based extraction.
        """
        text_parts = []
        
        for i, df in enumerate(tables, start=1):
            if df.empty:
                continue
            
            text_parts.append(f"[Table {i}]")
            
            # Add headers
            headers = ' | '.join(str(h) for h in df.columns)
            text_parts.append(headers)
            text_parts.append('-' * len(headers))
            
            # Add rows
            for _, row in df.iterrows():
                row_text = ' | '.join(str(v) for v in row.values)
                text_parts.append(row_text)
            
            text_parts.append('')  # Blank line between tables
        
        return '\n'.join(text_parts)


def extract_tables_simple(pdf_path: Path) -> list[pd.DataFrame]:
    """
    Simple convenience function for basic table extraction.
    """
    extractor = TableExtractor()
    return extractor.extract_tables(pdf_path)
