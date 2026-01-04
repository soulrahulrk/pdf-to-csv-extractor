"""
PDF to CSV - Main Entry Point

This is the main orchestration module that ties together all extraction,
parsing, and output components. It provides both a CLI interface and
a programmatic API for PDF field extraction.

Architecture Overview:
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                     EXTRACTION LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Text Layer  │  │    OCR      │  │   Tables    │           │
│  │ (pdfplumber)│  │ (Tesseract) │  │  (Camelot)  │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         └────────────────┼────────────────┘                   │
│                          ▼                                    │
│              ┌──────────────────────┐                        │
│              │   Merged Raw Text    │                        │
│              └──────────┬───────────┘                        │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      PARSING LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │Field Mapper │──│  Validator  │──│ Normalizer  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                             │
│                  ┌─────────────┐                             │
│                  │ CSV Writer  │                             │
│                  └─────────────┘                             │
└──────────────────────────────────────────────────────────────┘
"""

import sys
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

import click
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Local imports
from extractor import (
    PDFTextExtractor,
    OCRExtractor,
    TableExtractor,
    ExtractionResult,
    ExtractionMethod,
    remove_noise,
    merge_extraction_results,
    get_pdf_info,
    check_tesseract_installed,
)
from parser import (
    FieldMapper,
    ConfigLoader,
    FieldValidator,
    InvoiceFieldNormalizer,
)
from output import (
    CSVWriter,
    CSVConfig,
    BatchCSVWriter,
)


# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[Path] = None):
    """Configure loguru logging."""
    # Remove default handler
    logger.remove()
    
    # Console logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=log_level,
        colorize=True
    )
    
    # File logging
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB"
        )


@dataclass
class ExtractionConfig:
    """Configuration for the extraction pipeline."""
    
    # Input/output
    config_path: Path = Path("config/fields.yaml")
    
    # Extraction options
    enable_ocr: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    
    # Table extraction
    extract_tables: bool = True
    
    # Processing options
    confidence_threshold: float = 0.6
    include_line_items: bool = True
    
    # Output options
    output_format: str = "csv"
    include_metadata: bool = True
    include_confidence: bool = False


@dataclass
class ExtractionReport:
    """Report of extraction results for a single document."""
    
    source_file: str
    success: bool
    fields_extracted: dict = field(default_factory=dict)
    line_items: list = field(default_factory=list)
    confidence_scores: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    extraction_method: str = "unknown"
    processing_time_ms: int = 0
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            'source_file': self.source_file,
            'success': self.success,
            'fields_extracted': self.fields_extracted,
            'line_items': self.line_items,
            'confidence_scores': self.confidence_scores,
            'warnings': self.warnings,
            'errors': self.errors,
            'extraction_method': self.extraction_method,
            'processing_time_ms': self.processing_time_ms,
        }


class PDFExtractor:
    """
    Main extraction orchestrator.
    
    This class coordinates the entire extraction pipeline:
    1. PDF content extraction (text/OCR/tables)
    2. Field mapping and extraction
    3. Validation and normalization
    4. Output generation
    
    Usage:
        extractor = PDFExtractor(config_path=Path("config/fields.yaml"))
        result = extractor.process_pdf(Path("invoice.pdf"))
        print(result.fields_extracted)
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the PDF extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize extraction components."""
        # Load field configuration
        self.config_loader = ConfigLoader()
        if self.config.config_path.exists():
            self.config_loader.load(self.config.config_path)
        else:
            logger.warning(f"Config file not found: {self.config.config_path}")
        
        # Text extractor (always available)
        self.text_extractor = PDFTextExtractor()
        
        # OCR extractor (optional)
        self.ocr_extractor = None
        if self.config.enable_ocr:
            try:
                if check_tesseract_installed():
                    self.ocr_extractor = OCRExtractor(
                        language=self.config.ocr_language,
                        dpi=self.config.ocr_dpi
                    )
                else:
                    logger.warning("Tesseract not installed - OCR disabled")
            except Exception as e:
                logger.warning(f"OCR initialization failed: {e}")
        
        # Table extractor
        self.table_extractor = None
        if self.config.extract_tables:
            try:
                self.table_extractor = TableExtractor()
            except Exception as e:
                logger.warning(f"Table extractor initialization failed: {e}")
        
        # Field mapper
        self.field_mapper = FieldMapper(config=self.config_loader.config)
        
        # Validator
        self.validator = FieldValidator(
            date_formats=self.config_loader.get_date_formats()
        )
        
        # Normalizer
        self.normalizer = InvoiceFieldNormalizer(
            date_formats=self.config_loader.get_date_formats()
        )
    
    def process_pdf(self, pdf_path: Path) -> ExtractionReport:
        """
        Process a single PDF file.
        
        This is the main entry point for extraction.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractionReport with results
        """
        start_time = datetime.now()
        report = ExtractionReport(source_file=str(pdf_path))
        
        # Validate input
        pdf_info = get_pdf_info(pdf_path)
        if not pdf_info['readable']:
            report.errors.append(pdf_info.get('error', 'Cannot read PDF file'))
            return report
        
        logger.info(f"Processing: {pdf_path.name} ({pdf_info.get('size_mb', 0)} MB)")
        
        try:
            # Step 1: Extract raw content
            extraction_result = self._extract_content(pdf_path)
            report.extraction_method = extraction_result.method.value
            
            if not extraction_result.text.strip():
                report.errors.append("No text could be extracted from PDF")
                return report
            
            # Collect extraction warnings
            report.warnings.extend(extraction_result.warnings)
            
            # Step 2: Remove noise from extracted text
            cleaned_text = remove_noise(
                extraction_result.text,
                self.config_loader.noise_patterns
            )
            
            # Step 3: Extract fields
            extracted_fields = self.field_mapper.extract_fields(cleaned_text)
            
            # Step 4: Validate and normalize
            for field_result in extracted_fields:
                if not field_result.is_valid:
                    continue
                
                # Get field definition
                field_def = self.config_loader.get_field(field_result.name)
                field_type = field_def.field_type if field_def else 'string'
                
                # Validate
                validation = self.validator.validate_field(
                    field_result.name,
                    field_result.value,
                    field_type
                )
                
                if validation.is_valid:
                    # Normalize
                    normalized_value = self.normalizer.normalize_field(
                        validation.validated_value,
                        field_type,
                        field_result.name
                    )
                    
                    report.fields_extracted[field_result.name] = normalized_value
                    report.confidence_scores[field_result.name] = round(
                        field_result.confidence + validation.confidence_adjustment, 3
                    )
                else:
                    report.warnings.extend(
                        [f"{field_result.name}: {e}" for e in validation.errors]
                    )
            
            # Step 5: Extract line items (if configured)
            if self.config.include_line_items and self.table_extractor:
                line_items = self._extract_line_items(pdf_path)
                report.line_items = line_items
            
            # Step 6: Cross-validation
            cross_warnings = self.validator.cross_validate(report.fields_extracted)
            report.warnings.extend(cross_warnings)
            
            # Check required fields
            missing_required = []
            for field_def in self.config_loader.get_required_fields():
                if field_def.name not in report.fields_extracted:
                    missing_required.append(field_def.display_name)
            
            if missing_required:
                report.warnings.append(
                    f"Missing required fields: {', '.join(missing_required)}"
                )
            
            report.success = len(report.fields_extracted) > 0
            
        except Exception as e:
            logger.exception(f"Extraction failed: {e}")
            report.errors.append(str(e))
        
        finally:
            # Calculate processing time
            elapsed = datetime.now() - start_time
            report.processing_time_ms = int(elapsed.total_seconds() * 1000)
        
        return report
    
    def _extract_content(self, pdf_path: Path) -> ExtractionResult:
        """
        Extract content using layered strategy.
        
        Strategy:
        1. Try text layer extraction
        2. Analyze pages for OCR needs
        3. Apply OCR to pages with insufficient text
        4. Merge results
        """
        results = []
        
        # Step 1: Try text layer extraction
        logger.debug("Attempting text layer extraction...")
        text_result = self.text_extractor.extract_text(pdf_path)
        
        # Analyze pages
        page_analyses = self.text_extractor.analyze_pages(pdf_path)
        
        # Step 2: Determine if OCR is needed
        ocr_pages = []
        text_threshold = self.config_loader.settings.get('ocr_text_threshold', 50)
        
        for analysis in page_analyses:
            if analysis.needs_ocr(text_threshold):
                ocr_pages.append(analysis.page_number)
        
        # Step 3: Apply OCR if needed
        if ocr_pages and self.ocr_extractor:
            logger.info(f"Applying OCR to {len(ocr_pages)} pages...")
            
            for page_num in ocr_pages:
                try:
                    ocr_result = self.ocr_extractor.extract_from_pdf_page(
                        pdf_path, page_num
                    )
                    if ocr_result.text.strip():
                        results.append(ocr_result)
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
        
        # Add text layer result if it has content
        if text_result.text.strip():
            results.append(text_result)
        
        # Step 4: Merge results
        if not results:
            # Nothing extracted
            return ExtractionResult(
                text="",
                method=ExtractionMethod.UNKNOWN,
                confidence=0.0,
                warnings=["No content could be extracted"]
            )
        
        return merge_extraction_results(results)
    
    def _extract_line_items(self, pdf_path: Path) -> list[dict]:
        """Extract line items from tables in the PDF."""
        if not self.table_extractor:
            return []
        
        try:
            # Get column mappings from config
            line_items_config = self.config_loader.line_items_config
            
            if not line_items_config.get('enabled', False):
                return []
            
            column_mappings = line_items_config.get('column_mappings', {})
            
            if column_mappings:
                items = self.table_extractor.extract_line_items(
                    pdf_path, column_mappings
                )
            else:
                # Fall back to simple table extraction
                tables = self.table_extractor.extract_tables(pdf_path)
                items = []
                for df in tables:
                    items.extend(df.to_dict('records'))
            
            return items
            
        except Exception as e:
            logger.warning(f"Line item extraction failed: {e}")
            return []
    
    def process_directory(
        self,
        input_dir: Path,
        output_path: Path,
        pattern: str = "*.pdf"
    ) -> list[ExtractionReport]:
        """
        Process all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_path: Output CSV file path
            pattern: Glob pattern for PDF files
            
        Returns:
            List of ExtractionReport for each document
        """
        pdf_files = list(input_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        reports = []
        all_records = []
        
        # Get column names from config
        columns = [f.name for f in self.config_loader.fields]
        field_types = {f.name: f.field_type for f in self.config_loader.fields}
        
        console = Console()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing PDFs...", total=len(pdf_files))
            
            for pdf_path in pdf_files:
                progress.update(task, description=f"Processing {pdf_path.name}...")
                
                report = self.process_pdf(pdf_path)
                reports.append(report)
                
                if report.success:
                    record = report.fields_extracted.copy()
                    record['_source_file'] = pdf_path.name
                    record['_extraction_confidence'] = sum(
                        report.confidence_scores.values()
                    ) / max(len(report.confidence_scores), 1)
                    all_records.append(record)
                    
                    # Add line items if present
                    if report.line_items:
                        for item in report.line_items:
                            item_record = record.copy()
                            item_record.update(item)
                            all_records.append(item_record)
                
                progress.advance(task)
        
        # Write to CSV
        if all_records:
            writer = CSVWriter(
                output_path=output_path,
                columns=columns + ['_source_file', '_extraction_confidence'],
                field_types=field_types
            )
            writer.write_records(all_records)
            
            logger.info(f"Results written to {output_path}")
        else:
            logger.warning("No records extracted from any PDF")
        
        # Print summary
        self._print_summary(reports, console)
        
        return reports
    
    def _print_summary(self, reports: list[ExtractionReport], console: Console):
        """Print a summary table of extraction results."""
        table = Table(title="Extraction Summary")
        
        table.add_column("File", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Fields", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Time (ms)", justify="right")
        
        for report in reports:
            status = "[green]✓" if report.success else "[red]✗"
            field_count = len(report.fields_extracted)
            
            avg_confidence = ""
            if report.confidence_scores:
                avg = sum(report.confidence_scores.values()) / len(report.confidence_scores)
                avg_confidence = f"{avg:.0%}"
            
            # Truncate filename if too long
            filename = Path(report.source_file).name
            if len(filename) > 30:
                filename = filename[:27] + "..."
            
            table.add_row(
                filename,
                status,
                str(field_count),
                avg_confidence,
                str(report.processing_time_ms)
            )
        
        console.print()
        console.print(table)
        
        # Summary stats
        total = len(reports)
        successful = sum(1 for r in reports if r.success)
        
        console.print()
        console.print(f"[bold]Total:[/] {total} files")
        console.print(f"[bold green]Successful:[/] {successful}")
        console.print(f"[bold red]Failed:[/] {total - successful}")


# CLI Interface
@click.command()
@click.option(
    '--input', '-i',
    'input_path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Input PDF file or directory'
)
@click.option(
    '--output', '-o',
    'output_path',
    type=click.Path(path_type=Path),
    required=True,
    help='Output CSV file path'
)
@click.option(
    '--config', '-c',
    'config_path',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to fields.yaml configuration file'
)
@click.option(
    '--ocr/--no-ocr',
    'enable_ocr',
    default=True,
    help='Enable/disable OCR for scanned pages'
)
@click.option(
    '--ocr-language',
    default='eng',
    help='Tesseract language code (e.g., eng, fra, deu)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--log-file',
    type=click.Path(path_type=Path),
    default=None,
    help='Write logs to file'
)
@click.option(
    '--json-report',
    type=click.Path(path_type=Path),
    default=None,
    help='Write detailed JSON report'
)
def main(
    input_path: Path,
    output_path: Path,
    config_path: Optional[Path],
    enable_ocr: bool,
    ocr_language: str,
    verbose: bool,
    log_file: Optional[Path],
    json_report: Optional[Path]
):
    """
    PDF to CSV - Extract structured data from PDF documents.
    
    Examples:
    
        # Process single PDF
        python main.py -i invoice.pdf -o output.csv
        
        # Process directory of PDFs
        python main.py -i ./pdfs/ -o ./output/data.csv
        
        # With custom config and OCR
        python main.py -i ./pdfs/ -o output.csv -c ./config/fields.yaml --ocr
        
        # Disable OCR for faster processing
        python main.py -i ./pdfs/ -o output.csv --no-ocr
    """
    # Setup logging
    setup_logging(verbose=verbose, log_file=log_file)
    
    console = Console()
    console.print("[bold blue]PDF to CSV Extractor[/]")
    console.print()
    
    # Determine config path
    if config_path is None:
        # Try default location
        default_config = Path(__file__).parent / "config" / "fields.yaml"
        if default_config.exists():
            config_path = default_config
        else:
            console.print("[yellow]Warning: No config file specified and default not found[/]")
            console.print("Using built-in defaults")
    
    # Create extraction config
    extraction_config = ExtractionConfig(
        config_path=config_path if config_path else Path("config/fields.yaml"),
        enable_ocr=enable_ocr,
        ocr_language=ocr_language,
    )
    
    # Initialize extractor
    try:
        extractor = PDFExtractor(config=extraction_config)
    except Exception as e:
        console.print(f"[bold red]Initialization failed: {e}[/]")
        raise SystemExit(1)
    
    # Process input
    try:
        if input_path.is_file():
            # Single file
            console.print(f"Processing: {input_path.name}")
            report = extractor.process_pdf(input_path)
            
            if report.success:
                # Write to CSV
                record = report.fields_extracted.copy()
                record['_source_file'] = input_path.name
                
                columns = [f.name for f in extractor.config_loader.fields]
                field_types = {f.name: f.field_type for f in extractor.config_loader.fields}
                
                writer = CSVWriter(
                    output_path=output_path,
                    columns=columns + ['_source_file'],
                    field_types=field_types
                )
                writer.write_records([record])
                
                console.print(f"[green]✓ Output written to: {output_path}[/]")
                
                # Print extracted fields
                console.print("\n[bold]Extracted Fields:[/]")
                for name, value in report.fields_extracted.items():
                    confidence = report.confidence_scores.get(name, 0)
                    console.print(f"  {name}: {value} ({confidence:.0%})")
            else:
                console.print(f"[red]✗ Extraction failed[/]")
                for error in report.errors:
                    console.print(f"  Error: {error}")
            
            # Write JSON report if requested
            if json_report:
                with open(json_report, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2, default=str)
                console.print(f"Report written to: {json_report}")
            
        else:
            # Directory
            reports = extractor.process_directory(
                input_path,
                output_path,
                pattern="*.pdf"
            )
            
            # Write JSON report if requested
            if json_report:
                with open(json_report, 'w') as f:
                    json.dump([r.to_dict() for r in reports], f, indent=2, default=str)
                console.print(f"Report written to: {json_report}")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted[/]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/]")
        if verbose:
            logger.exception("Full traceback:")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
