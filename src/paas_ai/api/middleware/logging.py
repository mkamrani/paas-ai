"""
Logging middleware for API request/response tracking.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log API requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and log details."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Log request headers (excluding sensitive data)
        headers_to_log = {
            k: v for k, v in request.headers.items() 
            if k.lower() not in ['authorization', 'x-api-key', 'cookie']
        }
        if headers_to_log:
            logger.debug(f"[{request_id}] Request headers: {headers_to_log}")
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add custom headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.4f}"
            
            # Log response
            logger.info(
                f"[{request_id}] Response: {response.status_code} - "
                f"Time: {processing_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"[{request_id}] Request failed: {str(e)} - "
                f"Time: {processing_time:.4f}s",
                exc_info=True
            )
            
            # Re-raise the exception
            raise


def setup_request_logging(app) -> None:
    """
    Setup request logging middleware.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RequestLoggingMiddleware) 