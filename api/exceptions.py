from __future__ import annotations

from fastapi import HTTPException, status


class AppError(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Validation error"


class BrokerError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Broker communication error"


class MarketClosedError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Market is closed"


class InsufficientDataError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Insufficient data for analysis"


def to_http_exception(e: AppError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.detail)
