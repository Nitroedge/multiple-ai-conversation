"""
Enhanced Logging Configuration with Structured Logs and Correlation IDs

This module provides structured logging with correlation IDs for tracking requests
across the multi-agent conversation system.
"""

import logging
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from functools import wraps

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs JSON structured logs with correlation IDs
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(''),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
        if hasattr(record, 'agent_id'):
            log_entry["agent_id"] = record.agent_id
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'execution_time'):
            log_entry["execution_time_ms"] = record.execution_time
        if hasattr(record, 'component'):
            log_entry["component"] = record.component
        if hasattr(record, 'operation'):
            log_entry["operation"] = record.operation
        if hasattr(record, 'metadata'):
            log_entry["metadata"] = record.metadata

        return json.dumps(log_entry, ensure_ascii=False)


class CorrelationIdAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes correlation ID and additional context
    """

    def process(self, msg, kwargs):
        """Add correlation ID and context to log record"""
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = correlation_id.get('')
        kwargs['extra'] = extra
        return msg, kwargs

    def info_with_context(self, msg: str, **context):
        """Log info with additional context"""
        self.info(msg, extra=context)

    def error_with_context(self, msg: str, **context):
        """Log error with additional context"""
        self.error(msg, extra=context)

    def warning_with_context(self, msg: str, **context):
        """Log warning with additional context"""
        self.warning(msg, extra=context)

    def debug_with_context(self, msg: str, **context):
        """Log debug with additional context"""
        self.debug(msg, extra=context)


def setup_structured_logging(
    log_level: str = "INFO",
    enable_json_output: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_json_output: Whether to use JSON structured output
        log_file: Optional log file path
    """

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()

    if enable_json_output:
        console_handler.setFormatter(StructuredFormatter())
    else:
        # Fallback to standard format with correlation ID
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
            )
        )

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    logging.getLogger('motor').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())[:8]


def set_correlation_id(corr_id: str) -> None:
    """Set correlation ID for current context"""
    correlation_id.set(corr_id)


def get_correlation_id() -> str:
    """Get current correlation ID"""
    return correlation_id.get('')


def get_logger(name: str) -> CorrelationIdAdapter:
    """
    Get a structured logger with correlation ID support

    Args:
        name: Logger name (usually __name__)

    Returns:
        CorrelationIdAdapter instance
    """
    base_logger = logging.getLogger(name)
    return CorrelationIdAdapter(base_logger, {})


def log_execution_time(operation: str):
    """
    Decorator to log execution time of functions

    Args:
        operation: Description of the operation being timed
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                logger.info_with_context(
                    f"{operation} completed successfully",
                    operation=operation,
                    function=func.__name__,
                    execution_time=execution_time
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error_with_context(
                    f"{operation} failed: {str(e)}",
                    operation=operation,
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e)
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                logger.info_with_context(
                    f"{operation} completed successfully",
                    operation=operation,
                    function=func.__name__,
                    execution_time=execution_time
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error_with_context(
                    f"{operation} failed: {str(e)}",
                    operation=operation,
                    function=func.__name__,
                    execution_time=execution_time,
                    error=str(e)
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RequestLoggerMiddleware:
    """
    Middleware to add correlation IDs to FastAPI requests
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate correlation ID for request
            corr_id = generate_correlation_id()
            set_correlation_id(corr_id)

            # Log request start
            logger = get_logger(__name__)
            logger.info_with_context(
                f"Request started: {scope['method']} {scope['path']}",
                component="middleware",
                operation="request_start",
                method=scope['method'],
                path=scope['path'],
                correlation_id=corr_id
            )

        await self.app(scope, receive, send)


# Component-specific loggers
class ComponentLogger:
    """Specialized loggers for different system components"""

    @staticmethod
    def memory_logger() -> CorrelationIdAdapter:
        """Logger for memory management operations"""
        logger = get_logger("memory")
        return logger

    @staticmethod
    def agent_logger() -> CorrelationIdAdapter:
        """Logger for multi-agent coordination"""
        logger = get_logger("multi_agent")
        return logger

    @staticmethod
    def voice_logger() -> CorrelationIdAdapter:
        """Logger for voice processing"""
        logger = get_logger("voice")
        return logger

    @staticmethod
    def api_logger() -> CorrelationIdAdapter:
        """Logger for API operations"""
        logger = get_logger("api")
        return logger

    @staticmethod
    def websocket_logger() -> CorrelationIdAdapter:
        """Logger for WebSocket operations"""
        logger = get_logger("websocket")
        return logger


# Utility functions for common logging patterns
def log_memory_operation(operation: str, session_id: str, **context):
    """Log memory-related operations"""
    logger = ComponentLogger.memory_logger()
    logger.info_with_context(
        f"Memory operation: {operation}",
        component="memory",
        operation=operation,
        session_id=session_id,
        **context
    )


def log_agent_coordination(operation: str, agents: list, **context):
    """Log multi-agent coordination operations"""
    logger = ComponentLogger.agent_logger()
    logger.info_with_context(
        f"Agent coordination: {operation}",
        component="multi_agent",
        operation=operation,
        agents=agents,
        **context
    )


def log_api_request(endpoint: str, method: str, **context):
    """Log API request operations"""
    logger = ComponentLogger.api_logger()
    logger.info_with_context(
        f"API request: {method} {endpoint}",
        component="api",
        operation="api_request",
        endpoint=endpoint,
        method=method,
        **context
    )