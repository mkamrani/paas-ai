"""
Main FastAPI application for PaaS AI.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import ErrorResponse, HealthStatus
from .middleware import setup_cors, setup_request_logging, setup_security
from .routers import agent
from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 PaaS AI API starting up...")
    
    # Load configuration and validate system components
    try:
        from paas_ai.core.config import load_config
        config = load_config()
        logger.info(f"✅ Configuration loaded with {config.embedding.type} embeddings")
        
        # Test embeddings initialization
        from paas_ai.core.rag.embeddings import EmbeddingsFactory
        embeddings = EmbeddingsFactory.create_embeddings(config.embedding)
        logger.info("✅ Embeddings initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system components: {e}")
        # Don't prevent startup, but log the error
    
    logger.info("🎯 PaaS AI API ready to serve requests!")
    
    yield
    
    # Shutdown
    logger.info("🛑 PaaS AI API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="PaaS AI API",
        description="""
        **PaaS AI API** - Intelligent Platform-as-a-Service with RAG capabilities
        
        This API provides:
        - 🤖 **Agent endpoints** for intelligent Q&A using your knowledge base
        - 📚 **RAG endpoints** for managing and searching documents  
        - 🔧 **Configuration management** with profile support
        - 🔐 **Security** with API key authentication and rate limiting
        
        ## Authentication
        
        Most endpoints require an API key. You can authenticate using:
        - **Header**: `X-API-Key: your-api-key`
        - **Bearer token**: `Authorization: Bearer your-api-key`
        
        ## Configuration Profiles
        
        Override the default configuration profile using:
        - **Header**: `X-Config-Profile: local`
        - **Request body**: `{"config_profile": "local"}`
        
        Available profiles: `default`, `local`, `production`
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Setup middleware (order matters!)
    setup_cors(app)
    setup_request_logging(app)
    setup_security(app, 
                  enable_auth=os.getenv("DISABLE_API_AUTH", "false").lower() != "true",
                  enable_rate_limiting=os.getenv("DISABLE_RATE_LIMITING", "false").lower() != "true")
    
    # Register routers
    app.include_router(agent.router, prefix="/api/v1")
    # TODO: Add RAG router
    # app.include_router(rag.router, prefix="/api/v1")
    
    return app


# Create the application instance
app = create_app()


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthStatus,
    tags=["health"],
    summary="Health check",
    description="Check the health status of the API and its components."
)
async def health_check() -> HealthStatus:
    """Health check endpoint."""
    components = {}
    
    try:
        # Check configuration loading - use local profile for consistency
        from paas_ai.core.config import load_config
        config = load_config()  # Default to local profile
        components["config"] = "healthy"
        
        # Check embeddings
        from paas_ai.core.rag.embeddings import EmbeddingsFactory
        embeddings = EmbeddingsFactory.create_embeddings(config.embedding)
        components["embeddings"] = "healthy"
        
        # Check vectorstore (if exists)
        from paas_ai.core.rag.vectorstore import VectorStoreFactory
        try:
            vectorstore = VectorStoreFactory.load_vectorstore(config.vectorstore, embeddings)
            components["vectorstore"] = "healthy" if vectorstore else "no_data"
        except Exception:
            components["vectorstore"] = "error"
        
        # Check LLM (basic validation)
        components["llm"] = "healthy" if config.llm.provider else "not_configured"
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        components["system"] = "error"
    
    return HealthStatus(
        version="1.0.0",
        components=components
    )


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    error_response = ErrorResponse(
        message=exc.detail,
        error_code=f"HTTP_{exc.status_code}",
        request_id=getattr(request.state, 'request_id', None)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions."""
    error_response = ErrorResponse(
        message=exc.detail,
        error_code=f"HTTP_{exc.status_code}",
        request_id=getattr(request.state, 'request_id', None)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    error_response = ErrorResponse(
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        details={"errors": exc.errors()},
        request_id=getattr(request.state, 'request_id', None)
    )
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        message="Internal server error",
        error_code="INTERNAL_ERROR",
        request_id=getattr(request.state, 'request_id', None)
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# Root endpoint
@app.get(
    "/",
    tags=["root"],
    summary="API root",
    description="Get basic API information and links to documentation."
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🚀 Welcome to PaaS AI API",
        "version": "1.0.0",
        "description": "Intelligent Platform-as-a-Service with RAG capabilities",
        "docs": "/docs",
        "health": "/health",
        "api_version": "v1",
        "endpoints": {
            "agent": "/api/v1/agent",
            "rag": "/api/v1/rag"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting PaaS AI API on {host}:{port}")
    
    uvicorn.run(
        "src.paas_ai.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 