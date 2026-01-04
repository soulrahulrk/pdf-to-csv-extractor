"""
Retry Policies

Configurable retry logic with exponential backoff for handling transient failures.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional, List, Dict, Any, Callable, TypeVar, Type,
    Tuple, Union, Set
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


class RetryableError(Exception):
    """
    Exception that indicates a retryable failure.
    
    Use this to explicitly mark errors that should trigger retry.
    """
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class NonRetryableError(Exception):
    """
    Exception that should not be retried.
    
    Use this to explicitly prevent retry for certain failures.
    """
    pass


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    
    attempt_number: int
    error: Exception
    elapsed_time: float
    will_retry: bool
    next_delay: float


@dataclass
class RetryResult:
    """Result of a retried operation."""
    
    success: bool
    result: Any = None
    attempts: int = 0
    total_time: float = 0.0
    last_error: Optional[str] = None
    attempt_history: List[RetryAttempt] = field(default_factory=list)


class BackoffStrategy(Enum):
    """Backoff strategies for retry delays."""
    
    CONSTANT = auto()      # Same delay each time
    LINEAR = auto()        # Delay increases linearly
    EXPONENTIAL = auto()   # Delay doubles each time
    FIBONACCI = auto()     # Delay follows Fibonacci sequence


@dataclass
class ExponentialBackoff:
    """
    Exponential backoff configuration.
    
    Calculates delay as: base * (multiplier ^ attempt) + jitter
    """
    
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1  # Random jitter factor (0-1)
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for an attempt.
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


@dataclass
class RetryPolicy:
    """
    Configurable retry policy.
    
    Defines when and how to retry failed operations.
    """
    
    # Retry limits
    max_retries: int = 3
    max_time: Optional[float] = None  # Max total time for all retries
    
    # Delay configuration
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1
    
    # Exception handling
    retry_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )
    ignore_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {NonRetryableError, KeyboardInterrupt, SystemExit}
    )
    
    # Callbacks
    on_retry: Optional[Callable[[RetryAttempt], None]] = None
    should_retry: Optional[Callable[[Exception, int], bool]] = None
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for an attempt."""
        if self.backoff == BackoffStrategy.CONSTANT:
            delay = self.base_delay
        elif self.backoff == BackoffStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.backoff == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.multiplier ** attempt)
        elif self.backoff == BackoffStrategy.FIBONACCI:
            delay = self.base_delay * self._fibonacci(attempt + 1)
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def should_retry_exception(self, exc: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger retry.
        
        Args:
            exc: The exception that occurred
            attempt: Current attempt number
            
        Returns:
            Whether to retry
        """
        # Check ignore list first
        for ignore_type in self.ignore_exceptions:
            if isinstance(exc, ignore_type):
                return False
        
        # Check retry list
        for retry_type in self.retry_exceptions:
            if isinstance(exc, retry_type):
                # Custom callback check
                if self.should_retry:
                    return self.should_retry(exc, attempt)
                return True
        
        return False
    
    @classmethod
    def aggressive(cls) -> 'RetryPolicy':
        """Create an aggressive retry policy (many retries, short delays)."""
        return cls(
            max_retries=5,
            base_delay=0.5,
            max_delay=10.0,
            multiplier=1.5,
        )
    
    @classmethod
    def conservative(cls) -> 'RetryPolicy':
        """Create a conservative retry policy (few retries, longer delays)."""
        return cls(
            max_retries=2,
            base_delay=5.0,
            max_delay=60.0,
            multiplier=3.0,
        )
    
    @classmethod
    def no_retry(cls) -> 'RetryPolicy':
        """Create a policy with no retries."""
        return cls(max_retries=0)


def with_retry(
    policy: Optional[RetryPolicy] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable[[F], F]:
    """
    Decorator to add retry logic to a function.
    
    Args:
        policy: RetryPolicy to use
        max_retries: Maximum retry attempts (if policy not provided)
        base_delay: Base delay between retries (if policy not provided)
        
    Returns:
        Decorated function
        
    Usage:
        @with_retry(max_retries=3)
        def unreliable_operation():
            ...
        
        @with_retry(policy=RetryPolicy.aggressive())
        def another_operation():
            ...
    """
    if policy is None:
        policy = RetryPolicy(max_retries=max_retries, base_delay=base_delay)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return execute_with_retry(func, policy, *args, **kwargs)
        return wrapper  # type: ignore
    
    return decorator


def execute_with_retry(
    func: Callable[..., T],
    policy: RetryPolicy,
    *args,
    **kwargs,
) -> T:
    """
    Execute a function with retry logic.
    
    Args:
        func: Function to execute
        policy: Retry policy
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    start_time = time.time()
    last_error: Optional[Exception] = None
    
    for attempt in range(policy.max_retries + 1):
        try:
            return func(*args, **kwargs)
        
        except Exception as e:
            last_error = e
            elapsed = time.time() - start_time
            
            # Check if we should retry
            should_retry = (
                attempt < policy.max_retries and
                policy.should_retry_exception(e, attempt) and
                (policy.max_time is None or elapsed < policy.max_time)
            )
            
            # Calculate delay
            delay = policy.get_delay(attempt) if should_retry else 0
            
            # Create attempt record
            retry_attempt = RetryAttempt(
                attempt_number=attempt + 1,
                error=e,
                elapsed_time=elapsed,
                will_retry=should_retry,
                next_delay=delay,
            )
            
            # Callback
            if policy.on_retry:
                policy.on_retry(retry_attempt)
            
            if not should_retry:
                break
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            time.sleep(delay)
    
    # All retries exhausted
    if last_error:
        raise last_error
    
    raise RuntimeError("Retry logic error: no exception captured")


def retry_with_result(
    func: Callable[..., T],
    policy: RetryPolicy,
    *args,
    **kwargs,
) -> RetryResult:
    """
    Execute function with retry, returning detailed result.
    
    Args:
        func: Function to execute
        policy: Retry policy
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        RetryResult with success status and history
    """
    start_time = time.time()
    attempts = []
    result = None
    success = False
    
    for attempt in range(policy.max_retries + 1):
        try:
            result = func(*args, **kwargs)
            success = True
            break
        
        except Exception as e:
            elapsed = time.time() - start_time
            
            should_retry = (
                attempt < policy.max_retries and
                policy.should_retry_exception(e, attempt) and
                (policy.max_time is None or elapsed < policy.max_time)
            )
            
            delay = policy.get_delay(attempt) if should_retry else 0
            
            retry_attempt = RetryAttempt(
                attempt_number=attempt + 1,
                error=e,
                elapsed_time=elapsed,
                will_retry=should_retry,
                next_delay=delay,
            )
            attempts.append(retry_attempt)
            
            if policy.on_retry:
                policy.on_retry(retry_attempt)
            
            if not should_retry:
                break
            
            time.sleep(delay)
    
    return RetryResult(
        success=success,
        result=result,
        attempts=len(attempts) + (1 if success else 0),
        total_time=time.time() - start_time,
        last_error=str(attempts[-1].error) if attempts else None,
        attempt_history=attempts,
    )


class RetryContext:
    """
    Context manager for retry logic.
    
    Usage:
        with RetryContext(max_retries=3) as ctx:
            while ctx.should_continue:
                try:
                    result = risky_operation()
                    ctx.success()
                    break
                except Exception as e:
                    ctx.failed(e)
    """
    
    def __init__(self, policy: Optional[RetryPolicy] = None, **kwargs):
        """
        Initialize retry context.
        
        Args:
            policy: Retry policy to use
            **kwargs: Policy parameters if policy not provided
        """
        self.policy = policy or RetryPolicy(**kwargs)
        self._attempt = 0
        self._start_time = 0.0
        self._last_error: Optional[Exception] = None
        self._succeeded = False
    
    def __enter__(self) -> 'RetryContext':
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False
    
    @property
    def should_continue(self) -> bool:
        """Whether another attempt should be made."""
        if self._succeeded:
            return False
        
        if self._attempt > self.policy.max_retries:
            return False
        
        elapsed = time.time() - self._start_time
        if self.policy.max_time and elapsed >= self.policy.max_time:
            return False
        
        return True
    
    @property
    def attempt(self) -> int:
        """Current attempt number (1-indexed)."""
        return self._attempt
    
    def success(self) -> None:
        """Mark operation as successful."""
        self._succeeded = True
    
    def failed(self, error: Exception) -> None:
        """
        Record a failure and prepare for retry.
        
        Args:
            error: Exception that occurred
        """
        self._last_error = error
        self._attempt += 1
        
        if self.should_continue:
            delay = self.policy.get_delay(self._attempt - 1)
            logger.warning(
                f"Attempt {self._attempt} failed: {error}. "
                f"Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
    
    def raise_if_failed(self) -> None:
        """Raise last error if operation never succeeded."""
        if not self._succeeded and self._last_error:
            raise self._last_error


# Common retry-able operations
def retry_file_operation(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    **kwargs,
) -> T:
    """
    Retry a file operation with appropriate error handling.
    
    Handles common file errors like permission issues, file in use, etc.
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        retry_exceptions={
            IOError,
            OSError,
            PermissionError,
        },
        base_delay=0.5,
        backoff=BackoffStrategy.EXPONENTIAL,
    )
    
    return execute_with_retry(func, policy, *args, **kwargs)


def retry_network_operation(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    **kwargs,
) -> T:
    """
    Retry a network operation with appropriate error handling.
    
    Handles common network errors like timeouts, connection issues, etc.
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        retry_exceptions={
            ConnectionError,
            TimeoutError,
        },
        base_delay=1.0,
        max_delay=30.0,
        backoff=BackoffStrategy.EXPONENTIAL,
        jitter=0.2,
    )
    
    return execute_with_retry(func, policy, *args, **kwargs)
