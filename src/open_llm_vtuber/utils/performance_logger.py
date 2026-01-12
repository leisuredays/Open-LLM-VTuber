"""Performance logging utilities for measuring execution time of processes"""
import time
import functools
from typing import Optional, Callable, Any
from loguru import logger
from contextlib import contextmanager


class PerformanceTimer:
    """Context manager and decorator for measuring execution time"""

    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        logger.log(self.log_level, f"⏱️  [{self.name}] Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if exc_type is not None:
            logger.log(
                "ERROR",
                f"❌ [{self.name}] Failed after {self.elapsed:.3f}s - {exc_type.__name__}: {exc_val}"
            )
        else:
            logger.log(
                self.log_level,
                f"✅ [{self.name}] Completed in {self.elapsed:.3f}s"
            )
        return False


@contextmanager
def measure_time(name: str, log_level: str = "INFO"):
    """Context manager for measuring time"""
    timer = PerformanceTimer(name, log_level)
    with timer:
        yield timer


def log_execution_time(name: Optional[str] = None, log_level: str = "INFO"):
    """Decorator for measuring function execution time

    Usage:
        @log_execution_time("My Function")
        def my_function():
            pass

        @log_execution_time()
        async def async_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.log(log_level, f"⏱️  [{func_name}] Started")
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(
                    log_level,
                    f"✅ [{func_name}] Completed in {elapsed:.3f}s"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"❌ [{func_name}] Failed after {elapsed:.3f}s - {type(e).__name__}: {e}"
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            logger.log(log_level, f"⏱️  [{func_name}] Started")
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(
                    log_level,
                    f"✅ [{func_name}] Completed in {elapsed:.3f}s"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"❌ [{func_name}] Failed after {elapsed:.3f}s - {type(e).__name__}: {e}"
                )
                raise

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceMetrics:
    """Track and report performance metrics"""

    def __init__(self):
        self.metrics = {}

    def record(self, name: str, duration: float):
        """Record a timing measurement"""
        if name not in self.metrics:
            self.metrics[name] = {
                "count": 0,
                "total": 0.0,
                "min": float('inf'),
                "max": 0.0,
                "avg": 0.0
            }

        self.metrics[name]["count"] += 1
        self.metrics[name]["total"] += duration
        self.metrics[name]["min"] = min(self.metrics[name]["min"], duration)
        self.metrics[name]["max"] = max(self.metrics[name]["max"], duration)
        self.metrics[name]["avg"] = self.metrics[name]["total"] / self.metrics[name]["count"]

    def report(self):
        """Log performance report"""
        if not self.metrics:
            logger.info("📊 No performance metrics recorded")
            return

        logger.info("📊 Performance Report:")
        logger.info("=" * 80)
        for name, stats in sorted(self.metrics.items()):
            logger.info(
                f"  {name}:\n"
                f"    Count: {stats['count']}\n"
                f"    Total: {stats['total']:.3f}s\n"
                f"    Avg:   {stats['avg']:.3f}s\n"
                f"    Min:   {stats['min']:.3f}s\n"
                f"    Max:   {stats['max']:.3f}s"
            )
        logger.info("=" * 80)

    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()


# Global metrics instance
global_metrics = PerformanceMetrics()
