from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger


logger = get_logger(__name__)


class ErrorCode:
    """
    Application-level error codes.

    These codes are stable and easier for API clients to process
    than raw exception messages.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    INVALID_DATASET = "INVALID_DATASET"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_TRAINING_ERROR = "MODEL_TRAINING_ERROR"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


def build_error_response(
    code: str,
    message: str,
    details: dict | list | None = None,
) -> dict:
    """
    Build a standard API error response.

    All API errors should have the same JSON shape:
    - code
    - message
    - details
    """

    return {
        "code": code,
        "message": message,
        "details": details or {},
    }


def extract_http_error_data(exception: HTTPException) -> tuple[str, str, dict | list | None]:
    """
    Extract error code, message, and safe details from HTTPException.

    Important:
        We do not expose raw internal exception messages here.
        Endpoint code should pass only safe client-facing details.
    """

    if isinstance(exception.detail, dict):
        code = exception.detail.get("code", ErrorCode.HTTP_ERROR)
        message = exception.detail.get("message", "HTTP error occurred.")
        details = exception.detail.get("details", {})
        return code, message, details

    if isinstance(exception.detail, list):
        return ErrorCode.HTTP_ERROR, "HTTP error occurred.", exception.detail

    return ErrorCode.HTTP_ERROR, str(exception.detail), {}


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers for the FastAPI application.

    This keeps error responses consistent across the whole project.
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exception: HTTPException,
    ) -> JSONResponse:
        """
        Handle manually raised HTTPException errors.
        """

        code, message, details = extract_http_error_data(exception)

        return JSONResponse(
            status_code=exception.status_code,
            content=build_error_response(
                code=code,
                message=message,
                details=details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        """
        Handle request validation errors.

        These errors happen when the user sends invalid JSON,
        wrong field types, missing fields, or invalid enum values.
        """

        return JSONResponse(
            status_code=422,
            content=build_error_response(
                code=ErrorCode.VALIDATION_ERROR,
                message="Request validation failed.",
                details=exception.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exception: Exception,
    ) -> JSONResponse:
        """
        Handle unexpected server errors.

        Technical details are logged for developers,
        but they are not returned to API clients.
        """

        logger.exception("Unhandled internal server error.")

        return JSONResponse(
            status_code=500,
            content=build_error_response(
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                message="Unexpected internal server error.",
                details={},
            ),
        )