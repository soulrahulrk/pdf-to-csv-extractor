"""
Streaming Processor

Page-level streaming for memory-efficient processing of large documents.
Processes one page at a time to avoid loading entire documents into memory.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Iterator, Callable,
    TypeVar, Generic, Generator
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ProcessingResult:
    """Result from processing a single page or document."""
    
    success: bool
    page_number: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    memory_used_mb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_data(self) -> bool:
        return self.data is not None and len(self.data) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'page_number': self.page_number,
            'data': self.data,
            'error': self.error,
            'processing_time': round(self.processing_time, 3),
            'memory_used_mb': round(self.memory_used_mb, 2),
            'warnings': self.warnings,
        }


@dataclass
class BatchResult:
    """Result from processing a batch of documents."""
    
    total_documents: int
    successful: int
    failed: int
    total_pages: int
    total_time: float
    results: List[ProcessingResult]
    errors: Dict[str, str]  # filename -> error
    
    @property
    def success_rate(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return self.successful / self.total_documents
    
    @property
    def average_time_per_doc(self) -> float:
        if self.successful == 0:
            return 0.0
        return self.total_time / self.successful
    
    @property
    def pages_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_pages / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_documents': self.total_documents,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': round(self.success_rate, 3),
            'total_pages': self.total_pages,
            'total_time': round(self.total_time, 2),
            'average_time_per_doc': round(self.average_time_per_doc, 3),
            'pages_per_second': round(self.pages_per_second, 2),
            'errors': self.errors,
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming processor."""
    
    # Memory management
    max_memory_mb: int = 1024  # Max memory before forcing GC
    gc_interval: int = 10  # Force GC every N pages
    
    # Processing
    batch_size: int = 100  # Documents per batch
    max_pages_per_doc: Optional[int] = None  # Limit pages (None = all)
    
    # Timeouts
    page_timeout: float = 30.0  # Seconds per page
    document_timeout: float = 300.0  # Seconds per document
    
    # Error handling
    continue_on_error: bool = True
    max_consecutive_errors: int = 5
    
    # Progress
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    # Output
    output_dir: Optional[str] = None
    save_intermediate: bool = False


class StreamingProcessor:
    """
    Memory-efficient document processor using page-level streaming.
    
    Features:
    - Processes one page at a time
    - Automatic memory management and garbage collection
    - Progress tracking
    - Error recovery
    - Checkpointing for resumable processing
    
    Usage:
        processor = StreamingProcessor()
        
        # Stream pages from a document
        for result in processor.stream_pages(pdf_path):
            if result.success:
                process(result.data)
        
        # Process batch of documents
        batch_result = processor.process_batch(
            file_paths,
            extract_func=my_extractor,
        )
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize streaming processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or StreamingConfig()
        self._pages_processed = 0
        self._consecutive_errors = 0
        self._start_time = 0.0
    
    def stream_pages(
        self,
        pdf_path: str,
        extract_func: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
    ) -> Generator[ProcessingResult, None, None]:
        """
        Stream pages from a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            extract_func: Function to extract data from page
                         Signature: (page, page_number) -> dict
        
        Yields:
            ProcessingResult for each page
        """
        try:
            import pdfplumber
        except ImportError:
            yield ProcessingResult(
                success=False,
                error="pdfplumber not installed"
            )
            return
        
        self._start_time = time.time()
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                if self.config.max_pages_per_doc:
                    total_pages = min(total_pages, self.config.max_pages_per_doc)
                
                for page_num in range(total_pages):
                    result = self._process_page(
                        pdf,
                        page_num,
                        extract_func,
                    )
                    
                    yield result
                    
                    # Memory management
                    self._pages_processed += 1
                    if self._pages_processed % self.config.gc_interval == 0:
                        self._force_gc()
                    
                    # Error tracking
                    if result.success:
                        self._consecutive_errors = 0
                    else:
                        self._consecutive_errors += 1
                        
                        if self._consecutive_errors >= self.config.max_consecutive_errors:
                            yield ProcessingResult(
                                success=False,
                                error=f"Too many consecutive errors ({self._consecutive_errors})"
                            )
                            return
                    
                    # Progress callback
                    if self.config.progress_callback:
                        self.config.progress_callback(
                            page_num + 1,
                            total_pages,
                            os.path.basename(pdf_path)
                        )
        
        except Exception as e:
            logger.error(f"Error opening PDF {pdf_path}: {e}")
            yield ProcessingResult(
                success=False,
                error=str(e)
            )
    
    def _process_page(
        self,
        pdf: Any,
        page_num: int,
        extract_func: Optional[Callable],
    ) -> ProcessingResult:
        """Process a single page."""
        start_time = time.time()
        warnings = []
        
        try:
            page = pdf.pages[page_num]
            
            # Extract data
            if extract_func:
                data = extract_func(page, page_num)
            else:
                data = self._default_extract(page, page_num)
            
            processing_time = time.time() - start_time
            
            # Check timeout
            if processing_time > self.config.page_timeout:
                warnings.append(f"Page took {processing_time:.1f}s (timeout: {self.config.page_timeout}s)")
            
            return ProcessingResult(
                success=True,
                page_number=page_num,
                data=data,
                processing_time=processing_time,
                memory_used_mb=self._get_memory_usage(),
                warnings=warnings,
            )
        
        except Exception as e:
            logger.warning(f"Error processing page {page_num}: {e}")
            return ProcessingResult(
                success=False,
                page_number=page_num,
                error=str(e),
                processing_time=time.time() - start_time,
            )
    
    def _default_extract(self, page: Any, page_num: int) -> Dict[str, Any]:
        """Default extraction function."""
        return {
            'page_number': page_num,
            'text': page.extract_text() or '',
            'tables': [t.extract() for t in page.find_tables()],
            'width': page.width,
            'height': page.height,
        }
    
    def process_batch(
        self,
        file_paths: List[str],
        extract_func: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
        aggregate_func: Optional[Callable[[List[Dict]], Dict]] = None,
    ) -> BatchResult:
        """
        Process a batch of documents.
        
        Args:
            file_paths: List of PDF file paths
            extract_func: Function to extract data from page
            aggregate_func: Function to aggregate page results
            
        Returns:
            BatchResult with processing statistics
        """
        start_time = time.time()
        
        successful = 0
        failed = 0
        total_pages = 0
        all_results = []
        errors = {}
        
        for i, file_path in enumerate(file_paths):
            doc_start = time.time()
            page_results = []
            doc_success = True
            
            try:
                for result in self.stream_pages(file_path, extract_func):
                    page_results.append(result)
                    total_pages += 1
                    
                    if not result.success:
                        doc_success = False
                
                # Aggregate results
                if aggregate_func and page_results:
                    page_data = [r.data for r in page_results if r.has_data]
                    aggregated = aggregate_func(page_data)
                    
                    all_results.append(ProcessingResult(
                        success=doc_success,
                        data=aggregated,
                        processing_time=time.time() - doc_start,
                    ))
                else:
                    all_results.extend(page_results)
                
                if doc_success:
                    successful += 1
                else:
                    failed += 1
                    errors[os.path.basename(file_path)] = "Processing errors"
                    
            except Exception as e:
                failed += 1
                errors[os.path.basename(file_path)] = str(e)
                logger.error(f"Failed to process {file_path}: {e}")
            
            # Progress
            if self.config.progress_callback:
                self.config.progress_callback(
                    i + 1,
                    len(file_paths),
                    os.path.basename(file_path)
                )
            
            # Force GC between documents
            self._force_gc()
        
        return BatchResult(
            total_documents=len(file_paths),
            successful=successful,
            failed=failed,
            total_pages=total_pages,
            total_time=time.time() - start_time,
            results=all_results,
            errors=errors,
        )
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        recursive: bool = False,
        extract_func: Optional[Callable] = None,
    ) -> BatchResult:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern (glob)
            recursive: Whether to search recursively
            extract_func: Extraction function
            
        Returns:
            BatchResult
        """
        from pathlib import Path
        
        path = Path(directory)
        
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        file_paths = [str(f) for f in files if f.is_file()]
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        return self.process_batch(file_paths, extract_func)
    
    def _force_gc(self) -> None:
        """Force garbage collection."""
        gc.collect()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        current = self._get_memory_usage()
        if current > self.config.max_memory_mb:
            logger.warning(f"Memory usage ({current:.0f}MB) exceeds limit ({self.config.max_memory_mb}MB)")
            self._force_gc()
            return False
        return True


class PageIterator:
    """
    Iterator for streaming pages from multiple documents.
    
    Useful for processing pages across documents uniformly.
    """
    
    def __init__(
        self,
        file_paths: List[str],
        config: Optional[StreamingConfig] = None,
    ):
        """
        Initialize page iterator.
        
        Args:
            file_paths: List of PDF paths
            config: Processing configuration
        """
        self.file_paths = file_paths
        self.config = config or StreamingConfig()
        self._current_file_idx = 0
        self._current_pdf = None
        self._current_page_idx = 0
    
    def __iter__(self) -> Iterator[tuple]:
        """Iterate over all pages from all documents."""
        try:
            import pdfplumber
        except ImportError:
            return
        
        for file_path in self.file_paths:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        yield (file_path, page_num, page)
                        
                        # Memory management
                        if page_num % 10 == 0:
                            gc.collect()
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue


def create_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str,
) -> None:
    """
    Save processing checkpoint for resumption.
    
    Args:
        state: Processing state to save
        checkpoint_path: Path to checkpoint file
    """
    import json
    
    state['checkpoint_time'] = datetime.now().isoformat()
    
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load processing checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint state or None if not found
    """
    import json
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return state
    
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None
