"""
Document Intelligence Pipeline CLI

Command-line interface for the document intelligence pipeline.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
import time

from .pipeline import (
    DocumentIntelligencePipeline,
    PipelineConfig,
    ProcessingMode,
    DocumentResult,
)
from .doctypes.registry import (
    get_registry,
    list_document_types,
    detect_document_type,
)
from .doctypes.builtin_types import register_builtin_types


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.WARNING
    if verbose:
        level = logging.INFO
    if debug:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def find_pdfs(input_path: str, recursive: bool = False) -> List[str]:
    """Find PDF files in a path."""
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix.lower() == '.pdf':
            return [str(path)]
        else:
            raise ValueError(f"Not a PDF file: {input_path}")
    
    if path.is_dir():
        pattern = '**/*.pdf' if recursive else '*.pdf'
        return [str(p) for p in path.glob(pattern)]
    
    raise ValueError(f"Path not found: {input_path}")


def format_decision(decision: str) -> str:
    """Format decision for display."""
    colors = {
        'verified': '\033[92m',  # Green
        'likely': '\033[93m',     # Yellow
        'review_required': '\033[91m',  # Red
        'rejected': '\033[31m',   # Dark red
    }
    reset = '\033[0m'
    
    color = colors.get(decision, '')
    return f"{color}{decision.upper()}{reset}"


def print_result(result: DocumentResult, verbose: bool = False) -> None:
    """Print processing result to stdout."""
    print(f"\n{'='*60}")
    print(f"Document: {result.source_path}")
    print(f"Type: {result.document_type}")
    print(f"Pages: {result.page_count}")
    print(f"Decision: {format_decision(result.document_decision.decision.value)}")
    print(f"Confidence: {result.document_decision.overall_confidence:.2%}")
    print(f"Time: {result.processing_time_ms:.0f}ms")
    
    if result.fields:
        print(f"\nExtracted Fields ({len(result.fields)}):")
        for name, value in result.fields.items():
            decision = result.field_decisions.get(name)
            if decision:
                dec_str = format_decision(decision.decision.value)
                conf_str = f"{decision.confidence:.0%}"
                print(f"  {name}: {value} [{dec_str} {conf_str}]")
            else:
                print(f"  {name}: {value}")
    
    if verbose and result.validation_errors:
        print(f"\nValidation Errors:")
        for err in result.validation_errors:
            print(f"  ⚠ {err}")
    
    if verbose and result.arithmetic_errors:
        print(f"\nArithmetic Errors:")
        for err in result.arithmetic_errors:
            print(f"  ⚠ {err}")


def cmd_process(args: argparse.Namespace) -> int:
    """Process PDF documents."""
    
    # Find PDF files
    try:
        pdf_paths = find_pdfs(args.input, args.recursive)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not pdf_paths:
        print(f"No PDF files found in {args.input}", file=sys.stderr)
        return 1
    
    print(f"Found {len(pdf_paths)} PDF file(s)")
    
    # Create config
    mode = ProcessingMode(args.mode) if args.mode else ProcessingMode.STANDARD
    
    config = PipelineConfig(
        mode=mode,
        enable_ocr=not args.no_ocr,
        enable_table_detection=not args.no_tables,
        enable_semantic_validation=not args.no_validation,
        max_workers=args.workers,
        output_format=args.format,
    )
    
    # Create pipeline
    pipeline = DocumentIntelligencePipeline(config)
    
    # Process documents
    results: List[DocumentResult] = []
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        if not args.quiet:
            print(f"Processing [{i}/{len(pdf_paths)}]: {pdf_path}...")
        
        try:
            result = pipeline.process_document(pdf_path, args.type)
            results.append(result)
            
            if not args.quiet and not args.output:
                print_result(result, args.verbose)
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    total_time = time.time() - start_time
    metrics = pipeline.get_metrics()
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total documents: {metrics.total_documents}")
        print(f"Successful: {metrics.successful_documents}")
        print(f"Failed: {metrics.failed_documents}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg per document: {metrics.avg_time_per_doc_ms:.0f}ms")
        
        total_fields = (
            metrics.verified_fields + metrics.likely_fields +
            metrics.review_required_fields + metrics.rejected_fields
        )
        if total_fields > 0:
            print(f"\nDecision breakdown:")
            print(f"  Verified: {metrics.verified_fields} ({metrics.verified_fields/total_fields:.0%})")
            print(f"  Likely: {metrics.likely_fields} ({metrics.likely_fields/total_fields:.0%})")
            print(f"  Review Required: {metrics.review_required_fields} ({metrics.review_required_fields/total_fields:.0%})")
            print(f"  Rejected: {metrics.rejected_fields} ({metrics.rejected_fields/total_fields:.0%})")
    
    # Export results
    if args.output:
        pipeline.export_results(results, args.output, args.format)
        print(f"\nResults exported to: {args.output}")
    
    return 0 if metrics.failed_documents == 0 else 1


def cmd_list_types(args: argparse.Namespace) -> int:
    """List available document types."""
    
    register_builtin_types()
    types = list_document_types()
    
    print(f"\nAvailable Document Types ({len(types)}):")
    print("-" * 60)
    
    for type_info in types:
        print(f"\n{type_info['name']}")
        print(f"  Display: {type_info.get('display_name', 'N/A')}")
        print(f"  Description: {type_info.get('description', 'N/A')}")
        print(f"  Fields: {type_info.get('field_count', 0)}")
        print(f"  Category: {type_info.get('category', 'N/A')}")
    
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect document type from a PDF."""
    
    register_builtin_types()
    
    # Read sample text from PDF
    try:
        import pdfplumber
        with pdfplumber.open(args.input) as pdf:
            text = ""
            for page in pdf.pages[:3]:  # First 3 pages
                text += page.extract_text() or ""
    except ImportError:
        print("pdfplumber required for type detection", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading PDF: {e}", file=sys.stderr)
        return 1
    
    doc_type = detect_document_type(text)
    
    if doc_type:
        print(f"\nDetected Document Type: {doc_type.name}")
        print(f"Display Name: {doc_type.display_name}")
        print(f"Description: {doc_type.description}")
        print(f"\nExpected Fields ({len(doc_type.fields)}):")
        for field in doc_type.fields:
            req = " (required)" if field.required else ""
            print(f"  - {field.name}: {field.field_type.value}{req}")
    else:
        print("Could not detect document type")
        print("Use --type to specify document type manually")
    
    return 0


def cmd_export_schema(args: argparse.Namespace) -> int:
    """Export document type schema."""
    
    register_builtin_types()
    registry = get_registry()
    
    doc_type = registry.get(args.type)
    if not doc_type:
        print(f"Unknown document type: {args.type}", file=sys.stderr)
        return 1
    
    schema = doc_type.to_dict()
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)
        print(f"Schema exported to: {args.output}")
    else:
        print(json.dumps(schema, indent=2))
    
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate extracted data."""
    
    # Load data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    from .validation.semantic_rules import SemanticValidator
    from .validation.arithmetic_checks import ArithmeticChecker
    
    validator = SemanticValidator()
    checker = ArithmeticChecker()
    
    # Validate
    errors = []
    
    if isinstance(data, dict):
        for field_name, value in data.items():
            issues = validator.validate_field(field_name, str(value), 'text')
            errors.extend(issues)
        
        arith_issues = checker.check_all(data)
        errors.extend(arith_issues)
    
    if errors:
        print(f"\nValidation Errors ({len(errors)}):")
        for err in errors:
            print(f"  ⚠ {err}")
        return 1
    else:
        print("✓ Validation passed")
        return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='docint',
        description='Document Intelligence Pipeline - Extract structured data from PDFs',
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output',
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process PDF documents',
    )
    process_parser.add_argument(
        'input',
        help='PDF file or directory',
    )
    process_parser.add_argument(
        '-o', '--output',
        help='Output file path',
    )
    process_parser.add_argument(
        '-f', '--format',
        choices=['json', 'csv', 'review'],
        default='json',
        help='Output format',
    )
    process_parser.add_argument(
        '-t', '--type',
        help='Document type (auto-detect if not specified)',
    )
    process_parser.add_argument(
        '-m', '--mode',
        choices=['fast', 'standard', 'thorough', 'review'],
        default='standard',
        help='Processing mode',
    )
    process_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively',
    )
    process_parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of worker threads',
    )
    process_parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR',
    )
    process_parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Disable table detection',
    )
    process_parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation',
    )
    
    # List types command
    list_parser = subparsers.add_parser(
        'list-types',
        help='List available document types',
    )
    
    # Detect command
    detect_parser = subparsers.add_parser(
        'detect',
        help='Detect document type from PDF',
    )
    detect_parser.add_argument(
        'input',
        help='PDF file path',
    )
    
    # Export schema command
    schema_parser = subparsers.add_parser(
        'export-schema',
        help='Export document type schema',
    )
    schema_parser.add_argument(
        'type',
        help='Document type name',
    )
    schema_parser.add_argument(
        '-o', '--output',
        help='Output file path (stdout if not specified)',
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate extracted data',
    )
    validate_parser.add_argument(
        'input',
        help='JSON file with extracted data',
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    
    parser = create_parser()
    args = parser.parse_args(argv)
    
    setup_logging(args.verbose, args.debug)
    
    # Register built-in types
    register_builtin_types()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'process': cmd_process,
        'list-types': cmd_list_types,
        'detect': cmd_detect,
        'export-schema': cmd_export_schema,
        'validate': cmd_validate,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
