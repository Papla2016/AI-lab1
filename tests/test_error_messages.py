import pytest
from src.error_messages import describe_error

class FakeApiError(Exception):
    status_code = 429

def test_api_status_explains_quota_without_body():
    error = FakeApiError("secret-user-token")
    text = describe_error(error)
    assert "429" in text and "квот" in text.lower()
    assert "secret-user-token" not in text

def test_connection_error_without_exception_body():
    ConnectError = type("APIConnectionError", (Exception,), {})
    text = describe_error(ConnectError("https://service/key=secret"))
    assert "соединение" in text and "secret" not in text

def test_validation_error():
    assert "invalid" in describe_error(ValueError("invalid"))
