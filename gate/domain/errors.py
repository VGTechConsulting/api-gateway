"""Gate-generated errors.

Domain and application code raise these; only the HTTP adapter renders them
into the provider's error shape with an ``x-api-gateway-error`` header.
Messages never contain request content.
"""


class GateError(Exception):
    status: int = 500
    error_type: str = "api_error"
    gateway_code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        gateway_code: str | None = None,
        retry_after_s: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if gateway_code is not None:
            self.gateway_code = gateway_code
        self.retry_after_s = retry_after_s


class AuthenticationFailed(GateError):
    status = 401
    error_type = "authentication_error"
    gateway_code = "invalid_api_key"


class InvalidRequest(GateError):
    status = 400
    error_type = "invalid_request_error"
    gateway_code = "invalid_request"


class PermissionDenied(GateError):
    status = 403
    error_type = "permission_error"
    gateway_code = "permission_denied"


class NotFound(GateError):
    status = 404
    error_type = "not_found_error"
    gateway_code = "not_found"


class PayloadTooLarge(GateError):
    status = 413
    error_type = "request_too_large"
    gateway_code = "request_too_large"


class RateLimited(GateError):
    status = 429
    error_type = "rate_limit_error"
    gateway_code = "rate_limited"


class UpstreamConnectionFailed(GateError):
    status = 502
    error_type = "api_error"
    gateway_code = "upstream_connection_error"


class StorageUnavailable(GateError):
    status = 503
    error_type = "api_error"
    gateway_code = "storage_unavailable"


class UpstreamTimedOut(GateError):
    status = 504
    error_type = "api_error"
    gateway_code = "upstream_timeout"


class DetectorUnavailable(GateError):
    status = 503
    error_type = "api_error"
    gateway_code = "detector_error"


class ShuttingDown(GateError):
    status = 503
    error_type = "api_error"
    gateway_code = "shutting_down"


class ClientGone(GateError):
    """The client disconnected; nothing is delivered, the status is only recorded."""

    status = 499
    error_type = "api_error"
    gateway_code = "client_disconnected"


class InternalError(GateError):
    status = 500
    error_type = "api_error"
    gateway_code = "internal_error"
