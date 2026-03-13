from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class InvalidRequestError(AppError):
    def __init__(
        self,
        message: str = "Invalid request payload",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="invalid_request",
            message=message,
            status_code=400,
            details=details,
        )


class DownstreamServiceError(AppError):
    def __init__(
        self,
        service_name: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="downstream_service_error",
            message=message or f"Downstream service '{service_name}' failed",
            status_code=502,
            details={"service_name": service_name, **(details or {})},
        )


class ServiceUnavailableError(AppError):
    def __init__(
        self,
        service_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="service_unavailable",
            message=f"Service '{service_name}' is unavailable",
            status_code=503,
            details={"service_name": service_name, **(details or {})},
        )