"""
Performance and Scale Hardening

This package provides tools for processing large volumes of documents
efficiently with proper resource management, error handling, and recovery.
"""

from .streaming import (
    StreamingProcessor,
    StreamingConfig,
    ProcessingResult,
    BatchResult,
)
from .worker_pool import (
    WorkerPool,
    WorkerConfig,
    Task,
    TaskResult,
    TaskStatus,
)
from .retry import (
    RetryPolicy,
    RetryableError,
    with_retry,
    ExponentialBackoff,
)

__all__ = [
    'StreamingProcessor',
    'StreamingConfig',
    'ProcessingResult',
    'BatchResult',
    'WorkerPool',
    'WorkerConfig',
    'Task',
    'TaskResult',
    'TaskStatus',
    'RetryPolicy',
    'RetryableError',
    'with_retry',
    'ExponentialBackoff',
]
