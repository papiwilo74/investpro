"""JWT Authentication for the Inversion Helper API.

Single-user mode. Protege los endpoints del bot con token JWT (Bearer).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer()

# ── Configuración desde variables de entorno ──────────────────────────
_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))  # 8h

# Usuario único: leer de entorno o default para desarrollo
_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "papiwilo74")
_ADMIN_PASSWORD_HASH = os.environ.get(
    "ADMIN_PASSWORD_HASH",
    hashlib.sha256("blu301350".encode()).hexdigest(),
)


def verify_password(plain_password: str) -> bool:
    """Verifica contraseña contra hash SHA-256."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == _ADMIN_PASSWORD_HASH


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Crea un JWT token con expiración."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, _SECRET_KEY, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodifica y valida un JWT token."""
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Dependency que valida el token y retorna el username."""
    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido: falta subject",
        )
    return username


# ── Endpoints de autenticación ────────────────────────────────────────

def register_auth_routes(app) -> None:
    """Registra los endpoints de login en la app de FastAPI."""
    from api.schemas import LoginRequest, LoginResponse

    @app.post("/api/auth/login", response_model=LoginResponse)
    async def login(body: LoginRequest):
        if body.username != _ADMIN_USERNAME:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        if not verify_password(body.password):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        token = create_access_token({"sub": body.username})
        return LoginResponse(access_token=token)

    @app.get("/api/auth/verify")
    async def verify_token(username: str = Depends(get_current_user)):
        return {"status": "ok", "username": username}
