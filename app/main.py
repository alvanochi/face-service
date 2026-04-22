"""
Face Recognition Service — FastAPI application entrypoint.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health_router, router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import RequestIdMiddleware, setup_logging, get_logger
from app.services.face_pipeline import init_pipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    Startup:
        1. Configure structured logging.
        2. Initialize DB (create tables, enable pgvector).
        3. Load ML models into memory (once).

    Shutdown:
        Cleanup resources.
    """
    settings = get_settings()

    # --- Startup ---
    setup_logging()
    logger.info("startup.begin", env=settings.app_env)

    init_db()
    logger.info("startup.database_ready")

    init_pipeline()
    logger.info("startup.pipeline_ready")

    logger.info("startup.complete", port=settings.port)

    yield

    # --- Shutdown ---
    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()

    app = FastAPI(
        title="Face Recognition Service",
        description=(
            "Internal microservice for face detection, enrollment, "
            "verification (1:1), and recognition (1:N). "
            "Consumed by the Laravel authentication backend."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.openapi_enabled else None,
        redoc_url="/redoc" if settings.openapi_enabled else None,
        openapi_url="/openapi.json" if settings.openapi_enabled else None,
        lifespan=lifespan,
    )

    # --- Middleware ---
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Lock down in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(health_router)
    app.include_router(router)

    return app


app = create_app()
