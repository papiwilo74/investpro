import asyncio
import os
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from api.auth import register_auth_routes
from api.metrics import metrics_endpoint
from api.middleware import add_error_handlers, add_rate_limiting_middleware, add_security_headers_middleware
from api.routes import advisor, analysis, backtest, broker, market, ml, portfolio
from config import WATCHLIST, feature_flags, settings
from ml.ensemble import ensemble


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Gestión de ciclo de vida: validar settings, keepalive, detener bot al salir."""
    from config import settings

    _warnings = settings.validate()
    if _warnings:
        import logging

        for _w in _warnings:
            logging.warning(_w)

    from config import feature_flags

    app.state.feature_flags = feature_flags

    # Limpiar caché expirada al arrancar
    try:
        from data.cache_manager import cache_manager

        expired = cache_manager.clear_expired()
        if expired:
            logging.info("Limpieza de caché: %d entradas expiradas eliminadas", expired)
    except Exception:
        pass

    base_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("CLOUD_APP_URL")
    if base_url:
        _start_keepalive(base_url)

    yield

    from api.routes.broker import bot

    if bot.is_running:
        await bot.stop_async()


app = FastAPI(
    title="Inversion Helper API",
    version="2.0.0",
    description="API de trading automatizado con análisis técnico, ML y gestión de riesgo.\n\n"
    "## Modos de operación\n"
    "- **Broker**: Paper trading con Alpaca, gestión de posiciones y órdenes\n"
    "- **Machine Learning**: Predicción de tendencias, ensemble adaptativo\n"
    "- **Backtest**: Simulación histórica de estrategias\n"
    "- **Análisis**: Indicadores técnicos, señales compuestas\n"
    "- **Advisor**: Asistente online para decisiones de trading",
    summary="Inversion Helper - Trading bot API",
    contact={"name": "Inversion Helper", "url": "https://github.com/papiwilo74/investpro"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Registrar autenticación JWT
register_auth_routes(app)

# Middleware global
add_error_handlers(app)
add_security_headers_middleware(app)
add_rate_limiting_middleware(app, rpm=120)

# Habilitar CORS (restringido a origins conocidos en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        os.environ.get("RENDER_EXTERNAL_URL", ""),
    ],
    allow_credentials=False,  # True + allow_origins=* es inválido según spec
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Incluir Routers
app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(ml.router, prefix="/api/ml", tags=["Machine Learning"])
app.include_router(advisor.router, prefix="/api/advisor", tags=["Advisor"])
app.include_router(broker.router, prefix="/api/broker", tags=["Broker"])
app.include_router(broker.public_router, prefix="/api/broker", tags=["Broker Public"])


# Endpoint de Watchlist
@app.get("/api/watchlist", summary="Lista de tickers en watchlist")
async def get_watchlist():
    """Retorna los tickers monitoreados por el bot."""
    return WATCHLIST


# Feature flags
@app.get("/api/config/flags", summary="Feature flags activos")
async def get_feature_flags():
    """Retorna los feature flags según el ambiente actual."""
    return {"env": settings.ENV, "flags": feature_flags.to_dict()}


# Estado del caché de datos
@app.get("/api/data/info", summary="Estado de la capa de datos y caché")
async def get_data_info():
    """Métricas de la capa de datos: provider, caché, calidad."""
    from data.data_manager import data_manager

    dm = data_manager.health()
    from data.cache_manager import cache_manager

    dm["cache_index"] = cache_manager.to_dict()
    return dm


# Estado del Ensemble Adaptativo
@app.get("/api/ensemble/status", summary="Estado del ensemble adaptativo ML")
async def get_ensemble_status():
    """Métricas en vivo del AdaptiveEnsemble: pesos, agreement, accuracy por régimen."""
    from api.utils import sanitize_for_json

    return sanitize_for_json(ensemble.get_status())


# ── Frontend estático (build de React/Vite) ────────────────────────────
# Vite compila frontend/ -> api/static/ (ver frontend/vite.config.ts).
# En desarrollo se usa el dev server de Vite (puerto 3000) con proxy a /api.
_STATIC_BUILD = Path(_PROJECT_ROOT) / "api" / "static"
_FRONTEND_SRC = Path(_PROJECT_ROOT) / "frontend"
_STATIC_BUILD.mkdir(parents=True, exist_ok=True)


def _resolve_static_dir() -> Path:
    """Devuelve el directorio con el build de React si existe, si no, el source."""
    if (_STATIC_BUILD / "index.html").exists():
        return _STATIC_BUILD
    return _FRONTEND_SRC


_STATIC_DIR = _resolve_static_dir()


# Servir index.html en la raíz (sin cache para que los cambios se apliquen)
@app.get("/")
async def serve_index():
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Inversion Helper API is running. Run `npm run build` in frontend/ to generate the UI."}


@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.get("/performance", summary="Dashboard de performance")
async def serve_performance():
    perf_file = _FRONTEND_SRC / "performance.html"
    if perf_file.exists():
        return FileResponse(perf_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Performance page not found"}


# Montar /assets (build de Vite con nombres hashed) y /static (otros recursos)
if (_STATIC_BUILD / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_BUILD / "assets")), name="assets")
app.mount("/static", StaticFiles(directory=str(_FRONTEND_SRC)), name="static")


# Middleware para evitar cache del navegador en archivos JS/CSS (desarrollo)
@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── Live Monitoring ────────────────────────────────────────────────────
@app.get("/api/performance/live")
async def live_performance():
    """Payload combinado para el dashboard de monitoreo en vivo."""
    result: dict[str, object] = {
        "bot_status": "unknown",
        "metrics": {},
        "equity_curve": [],
        "active_positions": [],
        "recent_trades": [],
        "available": True,
    }

    try:
        from api.routes.broker import bot
        from api.utils import sanitize_for_json

        result["bot_status"] = "running" if bot.is_running else "stopped"
        if bot.perf_tracker is not None:
            result["metrics"] = sanitize_for_json(bot.perf_tracker.get_latest_metrics())
            result["equity_curve"] = sanitize_for_json(bot.perf_tracker.get_equity_curve(days=90))
        if hasattr(bot, "state_manager") and bot.state_manager is not None:
            result["active_positions"] = sanitize_for_json(bot.state_manager.get_positions())
    except Exception:
        result["available"] = False

    return result


# ── Keep-alive: auto-ping cada 10 min para que Render no duerma ────────
@app.get("/api/_ping", include_in_schema=False)
async def _keepalive_ping():
    """Endpoint interno para keepalive."""
    return {"ok": True}


def _start_keepalive(base_url: str) -> None:
    """Lanza background task que se auto-pinga para evitar que Render duerma el servicio."""

    async def _ping_loop():
        import urllib.request

        while True:
            try:
                await asyncio.sleep(600)  # 10 min
                urllib.request.urlopen(f"{base_url}/api/_ping", timeout=10)
            except Exception:
                pass

    asyncio.create_task(_ping_loop())


# ── Health check para la plataforma (Render/Fly.io) ────────────────────
@app.get("/health")
async def platform_health():
    """Health check real: verifica broker, bot, DB y data layer.

    No basta con responder 200 — el servicio puede estar vivo
    pero el bot colgado. Este endpoint verifica componentes críticos.
    """
    checks: dict[str, object] = {"api": "ok"}
    status_code = 200

    # Data layer health
    try:
        from data.data_manager import data_manager

        dm_health = data_manager.health()
        checks["data"] = dm_health
        if dm_health.get("quality_check", {}).get("status") != "ok":
            status_code = 503
    except Exception as e:
        checks["data"] = f"error: {e}"
        status_code = 503

    # Verificar broker
    try:
        from broker.alpaca_client import AlpacaClient

        c = AlpacaClient()
        connected = c.is_connected()
        checks["broker"] = "ok" if connected else "disconnected"
        if not connected:
            status_code = 503
    except Exception as e:
        checks["broker"] = f"error: {e}"
        status_code = 503

    # Verificar bot
    try:
        from api.routes.broker import bot

        checks["bot"] = "running" if bot.is_running else "stopped"
    except Exception:
        checks["bot"] = "unknown"

    from fastapi.responses import JSONResponse

    return JSONResponse(
        content={"status": "ok" if status_code == 200 else "degraded", "checks": checks},
        status_code=status_code,
    )


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Exposición de métricas Prometheus para scraping."""
    data, status, headers = metrics_endpoint()
    return Response(content=data, status_code=status, headers=headers)


# ── SPA fallback (DEBE ir al final, después de todas las rutas de la API) ──
@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """Catch-all para SPA: rutas no-API devuelven index.html (client-side routing)."""
    if path.startswith(("api/", "assets/", "static/", "docs", "redoc", "health", "metrics", "favicon")):
        return Response(status_code=404, content=b"Not Found", media_type="text/plain")
    candidate = _STATIC_DIR / path
    if candidate.is_file():
        return FileResponse(candidate)
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return Response(status_code=404, content=b"Not Found", media_type="text/plain")
