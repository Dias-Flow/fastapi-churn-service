from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dataset import router as dataset_router
from app.api.health import router as health_router
from app.api.model import router as model_router
from app.api.predict import router as predict_router
from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.errors import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.ml.persistence import load_churn_model, load_model_metadata
from app.ml.registry import clear_current_model, set_current_model


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
        The app tries to load a previously saved ML model from disk.

    If the model file is missing, corrupted, or incompatible,
    the application still starts without a loaded model.
    """

    try:
        model = load_churn_model()
        metadata = load_model_metadata()

        set_current_model(
            model=model,
            metadata=metadata,
        )

        logger.info("Saved churn model was loaded successfully.")

    except FileNotFoundError:
        clear_current_model()
        logger.info("No saved churn model found. Train the model using POST /model/train.")

    except Exception:
        clear_current_model()
        logger.exception(
            "Saved churn model could not be loaded. "
            "Application will start without a loaded model."
        )

    yield


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """

    setup_logging()

    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    @app.get(
        "/",
        tags=["Root"],
        summary="Service status",
    )
    def root():
        """
        Root endpoint.
        """

        return {"message": "ml churn service is running"}

    app.include_router(dataset_router)
    app.include_router(model_router)
    app.include_router(predict_router)
    app.include_router(health_router)

    return app


app = create_app()