import sys
import os
import asyncio
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import WATCHLIST
from api.routes import market, analysis, backtest, portfolio, ml, advisor, broker

app = FastAPI(title="Inversion Helper API", version="2.0.0")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir Routers
app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(ml.router, prefix="/api/ml", tags=["Machine Learning"])
app.include_router(advisor.router, prefix="/api/advisor", tags=["Advisor"])
app.include_router(broker.router, prefix="/api/broker", tags=["Broker"])

# Endpoint de Watchlist
@app.get("/api/watchlist")
async def get_watchlist():
    return WATCHLIST

# Configurar Frontend Estático si la carpeta existe
frontend_path = Path(_PROJECT_ROOT) / "frontend"
frontend_path.mkdir(parents=True, exist_ok=True)

# Servir index.html en la raíz (sin cache para que los cambios JS se apliquen)
@app.get("/")
async def serve_index():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Inversion Helper API is running. Place index.html in frontend/ to serve the UI."}

# Montar resto de recursos estáticos en /static
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Middleware para evitar cache del navegador en archivos JS/CSS (desarrollo)
@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── Keep-alive: auto-ping cada 10 min para que Render no duerma ────────
@app.on_event("startup")
async def _start_keepalive():
    """Background task que se auto-pinga para evitar que el servicio sleep."""
    base_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("CLOUD_APP_URL")

    if not base_url:
        # En local o sin URL configurada, no hacer nada
        return

    async def _ping_loop():
        import urllib.request
        while True:
            try:
                await asyncio.sleep(600)  # 10 min
                url = f"{base_url}/api/broker/health"
                urllib.request.urlopen(url, timeout=10)
            except Exception:
                pass

    asyncio.create_task(_ping_loop())


# ── Health check para la plataforma (Render/Fly.io) ────────────────────
@app.get("/health")
async def platform_health():
    """Health check simple para que la plataforma sepa que el servicio está vivo."""
    return {"status": "ok"}
