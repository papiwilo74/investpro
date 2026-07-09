"""Tests for JWT authentication module."""
import hashlib
import os
from datetime import datetime, timedelta
from importlib import reload

import pytest
from jose import jwt

import api.auth
from api.auth import (
    verify_password,
    create_access_token,
    decode_access_token,
    _SECRET_KEY,
    _ALGORITHM,
)


class TestPasswordVerification:
    def test_verify_correct_password(self):
        assert verify_password("inversion_helper_dev") is True

    def test_verify_wrong_password(self):
        assert verify_password("wrong_password") is False

    def test_verify_empty_password(self):
        assert verify_password("") is False

    def test_verify_with_env_override(self):
        custom_hash = hashlib.sha256("custom_pass".encode()).hexdigest()
        os.environ["ADMIN_PASSWORD_HASH"] = custom_hash
        reload(api.auth)
        from api.auth import verify_password as vp
        assert vp("custom_pass") is True
        assert vp("wrong") is False
        del os.environ["ADMIN_PASSWORD_HASH"]
        reload(api.auth)


class TestJWTToken:
    def test_create_token(self):
        token = create_access_token({"sub": "admin"})
        assert isinstance(token, str)
        assert len(token) > 20

    def test_decode_valid_token(self):
        token = create_access_token({"sub": "admin"})
        payload = decode_access_token(token)
        assert payload["sub"] == "admin"
        assert "exp" in payload

    def test_token_with_custom_claims(self):
        token = create_access_token({"sub": "admin", "role": "admin"})
        payload = decode_access_token(token)
        assert payload["sub"] == "admin"
        assert payload["role"] == "admin"

    def test_invalid_token_raises(self):
        with pytest.raises(Exception):
            decode_access_token("invalid.token.here")

    def test_expired_token_raises(self):
        token = jwt.encode(
            {"sub": "admin", "exp": datetime.utcnow() - timedelta(hours=1)},
            _SECRET_KEY,
            algorithm=_ALGORITHM,
        )
        with pytest.raises(Exception):
            decode_access_token(token)

    def test_token_with_different_secret_fails(self):
        token = jwt.encode(
            {"sub": "admin", "exp": datetime.utcnow() + timedelta(hours=1)},
            "wrong_secret",
            algorithm=_ALGORITHM,
        )
        with pytest.raises(Exception):
            decode_access_token(token)


class TestLoginEndpoint:
    def test_login_schema(self):
        from api.schemas import LoginRequest, LoginResponse
        req = LoginRequest(username="admin", password="test")
        assert req.username == "admin"
        assert req.password == "test"
        resp = LoginResponse(access_token="token123")
        assert resp.access_token == "token123"
        assert resp.token_type == "bearer"

    def test_auth_verify_response(self):
        from api.schemas import AuthVerifyResponse
        resp = AuthVerifyResponse(status="ok", username="admin")
        assert resp.status == "ok"
        assert resp.username == "admin"
