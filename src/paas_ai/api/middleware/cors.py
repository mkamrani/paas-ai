"""
CORS middleware configuration for the API.
"""

from fastapi.middleware.cors import CORSMiddleware
from typing import List


def setup_cors(app, allowed_origins: List[str] = None) -> None:
    """
    Setup CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins. If None, uses default development settings.
    """
    if allowed_origins is None:
        # Default development settings - restrict in production
        allowed_origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Config-Profile"
        ],
        expose_headers=[
            "X-Request-ID",
            "X-Processing-Time",
            "X-Rate-Limit-Remaining"
        ]
    ) 