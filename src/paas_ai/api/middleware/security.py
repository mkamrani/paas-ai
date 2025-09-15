"""
Security middleware for API authentication and rate limiting.
"""

import os
import time
from typing import Optional, Dict, Any
from collections import defaultdict, deque
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.api.security")


class APIKeyAuth(HTTPBearer):
    """API Key authentication scheme."""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        # Load API keys from environment
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> set:
        """Load valid API keys from environment variables."""
        api_keys = set()
        
        # Load primary API key
        primary_key = os.getenv("PAAS_AI_API_KEY")
        if primary_key:
            api_keys.add(primary_key)
        
        # Load additional API keys (comma-separated)
        additional_keys = os.getenv("PAAS_AI_API_KEYS")
        if additional_keys:
            api_keys.update(key.strip() for key in additional_keys.split(","))
        
        return api_keys
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key from request."""
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        
        # Also check Authorization header
        if not api_key:
            credentials: HTTPAuthorizationCredentials = await super().__call__(request)
            if credentials:
                api_key = credentials.credentials
        
        # If no API keys configured, allow all requests (development mode)
        if not self.api_keys:
            logger.warning("No API keys configured - allowing all requests")
            return "development"
        
        # Validate API key
        if not api_key or api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempted from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # In-memory storage for rate limiting (use Redis in production)
        self.minute_requests: Dict[str, deque] = defaultdict(deque)
        self.hour_requests: Dict[str, deque] = defaultdict(deque)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if available, otherwise IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}..."  # First 8 chars for logging
        
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _is_rate_limited(self, client_id: str) -> tuple[bool, Dict[str, Any]]:
        """Check if client is rate limited."""
        now = time.time()
        
        # Clean old entries and count current requests
        minute_window = now - 60
        hour_window = now - 3600
        
        # Minute-based rate limiting
        minute_queue = self.minute_requests[client_id]
        while minute_queue and minute_queue[0] < minute_window:
            minute_queue.popleft()
        
        # Hour-based rate limiting  
        hour_queue = self.hour_requests[client_id]
        while hour_queue and hour_queue[0] < hour_window:
            hour_queue.popleft()
        
        minute_count = len(minute_queue)
        hour_count = len(hour_queue)
        
        # Check limits
        minute_exceeded = minute_count >= self.requests_per_minute
        hour_exceeded = hour_count >= self.requests_per_hour
        
        rate_limit_info = {
            "requests_per_minute": minute_count,
            "requests_per_hour": hour_count,
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
            "minute_remaining": max(0, self.requests_per_minute - minute_count),
            "hour_remaining": max(0, self.requests_per_hour - hour_count),
            "reset_minute": int(now + (60 - (now % 60))),
            "reset_hour": int(now + (3600 - (now % 3600)))
        }
        
        return minute_exceeded or hour_exceeded, rate_limit_info
    
    def _record_request(self, client_id: str) -> None:
        """Record a request for rate limiting."""
        now = time.time()
        self.minute_requests[client_id].append(now)
        self.hour_requests[client_id].append(now)
    
    async def dispatch(self, request: Request, call_next) -> Any:
        """Check rate limits and process request."""
        client_id = self._get_client_id(request)
        
        # Check rate limits
        is_limited, rate_info = self._is_rate_limited(client_id)
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-Rate-Limit-Minute": str(rate_info["limit_per_minute"]),
                    "X-Rate-Limit-Hour": str(rate_info["limit_per_hour"]),
                    "X-Rate-Limit-Remaining-Minute": str(rate_info["minute_remaining"]),
                    "X-Rate-Limit-Remaining-Hour": str(rate_info["hour_remaining"]),
                    "X-Rate-Limit-Reset-Minute": str(rate_info["reset_minute"]),
                    "X-Rate-Limit-Reset-Hour": str(rate_info["reset_hour"]),
                    "Retry-After": "60"
                }
            )
        
        # Record the request
        self._record_request(client_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-Rate-Limit-Remaining-Minute"] = str(rate_info["minute_remaining"] - 1)
        response.headers["X-Rate-Limit-Remaining-Hour"] = str(rate_info["hour_remaining"] - 1)
        
        return response


def setup_security(app, enable_auth: bool = True, enable_rate_limiting: bool = True) -> None:
    """
    Setup security middleware.
    
    Args:
        app: FastAPI application instance
        enable_auth: Whether to enable API key authentication
        enable_rate_limiting: Whether to enable rate limiting
    """
    if enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
        logger.info("Rate limiting middleware enabled")
    
    # API key authentication is handled at the dependency level, not middleware
    if enable_auth:
        logger.info("API key authentication enabled") 