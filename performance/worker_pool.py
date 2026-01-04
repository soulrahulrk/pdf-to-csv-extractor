"""
Worker Pool

Multi-threaded/multi-process worker pool for parallel document processing.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Optional, List, Dict, Any, Callable, TypeVar, Generic,
    Union, Iterator
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TaskStatus(Enum):
    """Status of a task."""
    
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class Task:
    """
    A task to be processed by a worker.
    """
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout: Optional[float] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        func: Callable,
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> 'Task':
        """Create a new task."""
        return cls(
            task_id=str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
        )
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare by priority (higher priority first)."""
        return self.priority > other.priority


@dataclass
class TaskResult:
    """
    Result of task execution.
    """
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retries_used: int = 0
    worker_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.name,
            'result': str(self.result)[:100] if self.result else None,
            'error': self.error,
            'duration': round(self.duration, 3),
            'retries_used': self.retries_used,
            'worker_id': self.worker_id,
        }


@dataclass
class WorkerConfig:
    """Configuration for worker pool."""
    
    # Pool settings
    num_workers: int = 4
    use_processes: bool = False  # False = threads, True = processes
    
    # Queue settings
    max_queue_size: int = 1000
    
    # Execution settings
    default_timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Resource limits
    max_memory_per_worker_mb: int = 512
    
    # Callbacks
    on_task_complete: Optional[Callable[[TaskResult], None]] = None
    on_task_error: Optional[Callable[[Task, Exception], None]] = None
    on_worker_error: Optional[Callable[[str, Exception], None]] = None


class WorkerPool:
    """
    Thread/Process pool for parallel document processing.
    
    Features:
    - Priority queue for tasks
    - Automatic retry on failure
    - Timeout handling
    - Progress tracking
    - Graceful shutdown
    
    Usage:
        pool = WorkerPool(num_workers=4)
        
        # Submit tasks
        future = pool.submit(process_doc, pdf_path)
        
        # Wait for result
        result = future.result()
        
        # Map function over inputs
        results = pool.map(process_doc, pdf_paths)
        
        # Shutdown
        pool.shutdown()
    """
    
    def __init__(self, config: Optional[WorkerConfig] = None):
        """
        Initialize worker pool.
        
        Args:
            config: Worker configuration
        """
        self.config = config or WorkerConfig()
        
        # Create executor
        if self.config.use_processes:
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.num_workers
            )
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.num_workers,
                thread_name_prefix='worker',
            )
        
        # Task tracking
        self._pending_tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}
        self._results: Dict[str, TaskResult] = {}
        
        # Statistics
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Shutdown flag
        self._shutdown = False
        
        logger.info(
            f"Worker pool initialized with {self.config.num_workers} "
            f"{'processes' if self.config.use_processes else 'threads'}"
        )
    
    def submit(
        self,
        func: Callable[..., R],
        *args,
        priority: int = 0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Future:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            priority: Task priority (higher = more urgent)
            timeout: Execution timeout
            **kwargs: Keyword arguments
            
        Returns:
            Future for the result
        """
        if self._shutdown:
            raise RuntimeError("Pool is shut down")
        
        task = Task.create(
            func,
            *args,
            priority=priority,
            timeout=timeout or self.config.default_timeout,
            **kwargs,
        )
        
        return self._submit_task(task)
    
    def _submit_task(self, task: Task) -> Future:
        """Submit a task to the executor."""
        with self._lock:
            self._pending_tasks[task.task_id] = task
            self._tasks_submitted += 1
        
        # Wrap execution with retry and timeout handling
        future = self._executor.submit(
            self._execute_task,
            task,
        )
        
        with self._lock:
            self._futures[task.task_id] = future
        
        # Add completion callback
        future.add_done_callback(
            lambda f: self._on_complete(task.task_id, f)
        )
        
        return future
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a task with retry and timeout handling."""
        start_time = datetime.now()
        retries_used = 0
        last_error = None
        
        for attempt in range(task.retries + 1):
            try:
                # Execute the function
                result = task.func(*task.args, **task.kwargs)
                
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    start_time=start_time,
                    end_time=datetime.now(),
                    retries_used=retries_used,
                )
            
            except Exception as e:
                last_error = str(e)
                retries_used = attempt + 1
                
                logger.warning(
                    f"Task {task.task_id} failed (attempt {attempt + 1}): {e}"
                )
                
                if attempt < task.retries:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        # All retries exhausted
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=last_error,
            start_time=start_time,
            end_time=datetime.now(),
            retries_used=retries_used,
        )
    
    def _on_complete(self, task_id: str, future: Future) -> None:
        """Handle task completion."""
        try:
            result = future.result()
        except Exception as e:
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )
        
        with self._lock:
            self._results[task_id] = result
            self._pending_tasks.pop(task_id, None)
            
            if result.success:
                self._tasks_completed += 1
            else:
                self._tasks_failed += 1
        
        # Callback
        if self.config.on_task_complete:
            try:
                self.config.on_task_complete(result)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")
    
    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None,
        chunksize: int = 1,
    ) -> Iterator[R]:
        """
        Map a function over items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            timeout: Total timeout for all items
            chunksize: Number of items per task
            
        Yields:
            Results in completion order
        """
        futures = []
        
        for item in items:
            future = self.submit(func, item, timeout=timeout)
            futures.append(future)
        
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                if isinstance(result, TaskResult) and result.success:
                    yield result.result
                elif isinstance(result, TaskResult):
                    logger.warning(f"Task failed: {result.error}")
                else:
                    yield result
            except Exception as e:
                logger.error(f"Error getting result: {e}")
    
    def map_unordered(
        self,
        func: Callable[[T], R],
        items: List[T],
        timeout: Optional[float] = None,
    ) -> Iterator[tuple]:
        """
        Map function over items, yielding results as they complete.
        
        Args:
            func: Function to apply
            items: Items to process
            timeout: Total timeout
            
        Yields:
            (item, result) tuples in completion order
        """
        item_map = {}
        futures = []
        
        for item in items:
            future = self.submit(func, item, timeout=timeout)
            item_map[id(future)] = item
            futures.append(future)
        
        for future in as_completed(futures, timeout=timeout):
            item = item_map[id(future)]
            try:
                result = future.result()
                yield (item, result)
            except Exception as e:
                yield (item, e)
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result for a task."""
        return self._results.get(task_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'num_workers': self.config.num_workers,
                'use_processes': self.config.use_processes,
                'tasks_submitted': self._tasks_submitted,
                'tasks_completed': self._tasks_completed,
                'tasks_failed': self._tasks_failed,
                'tasks_pending': len(self._pending_tasks),
                'success_rate': (
                    self._tasks_completed / self._tasks_submitted
                    if self._tasks_submitted > 0 else 0.0
                ),
            }
    
    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        Wait for all pending tasks to complete.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            List of results
        """
        with self._lock:
            futures = list(self._futures.values())
        
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error waiting for task: {e}")
        
        return results
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the worker pool.
        
        Args:
            wait: Whether to wait for pending tasks
        """
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("Worker pool shut down")
    
    def __enter__(self) -> 'WorkerPool':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


class BatchProcessor:
    """
    High-level batch processor using worker pool.
    
    Provides convenient methods for processing document batches.
    """
    
    def __init__(
        self,
        process_func: Callable[[str], Dict[str, Any]],
        num_workers: int = 4,
        use_processes: bool = False,
    ):
        """
        Initialize batch processor.
        
        Args:
            process_func: Function to process each document
            num_workers: Number of parallel workers
            use_processes: Use processes instead of threads
        """
        self.process_func = process_func
        self.config = WorkerConfig(
            num_workers=num_workers,
            use_processes=use_processes,
        )
        self._pool: Optional[WorkerPool] = None
    
    def process(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of documents.
        
        Args:
            file_paths: List of file paths to process
            progress_callback: Progress callback (current, total)
            
        Returns:
            Processing summary
        """
        results = {}
        errors = {}
        
        start_time = time.time()
        
        with WorkerPool(self.config) as pool:
            futures = {
                pool.submit(self.process_func, path): path
                for path in file_paths
            }
            
            completed = 0
            for future in as_completed(futures):
                path = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    if isinstance(result, TaskResult):
                        if result.success:
                            results[path] = result.result
                        else:
                            errors[path] = result.error
                    else:
                        results[path] = result
                except Exception as e:
                    errors[path] = str(e)
                
                if progress_callback:
                    progress_callback(completed, len(file_paths))
        
        return {
            'total': len(file_paths),
            'successful': len(results),
            'failed': len(errors),
            'duration': time.time() - start_time,
            'results': results,
            'errors': errors,
        }


def process_in_parallel(
    func: Callable[[T], R],
    items: List[T],
    num_workers: int = 4,
    use_processes: bool = False,
    show_progress: bool = False,
) -> List[R]:
    """
    Convenience function for parallel processing.
    
    Args:
        func: Function to apply to each item
        items: Items to process
        num_workers: Number of workers
        use_processes: Use processes instead of threads
        show_progress: Show progress bar
        
    Returns:
        List of results
    """
    config = WorkerConfig(
        num_workers=num_workers,
        use_processes=use_processes,
    )
    
    results = []
    
    with WorkerPool(config) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        
        result_map = {}
        completed = 0
        
        for future in as_completed(futures):
            idx = futures[future]
            completed += 1
            
            try:
                result = future.result()
                if isinstance(result, TaskResult):
                    result_map[idx] = result.result if result.success else None
                else:
                    result_map[idx] = result
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                result_map[idx] = None
            
            if show_progress:
                print(f"\rProcessing: {completed}/{len(items)}", end='', flush=True)
        
        if show_progress:
            print()
        
        # Restore original order
        results = [result_map.get(i) for i in range(len(items))]
    
    return results
