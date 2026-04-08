"""
Entry point that exposes both the bias-detection API and claim-verification API
under separate prefixes on a single FastAPI app.

Run from repo root:
    uvicorn server.server_combined:app --reload --port 8000

Routes:
    /bias/...   -> existing bias detection endpoints (e.g., /bias/analyze, /bias/health)
    /claim/...  -> existing claim verification endpoints (e.g., /claim/verify-claims-from-passage)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import importlib.util

app = FastAPI(title="Veritas API (combined)")

# Broad CORS for local dev; tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bias_server = None
claim_server = None
loaded_bias = False
loaded_claim = False

# Mount existing apps under prefixes if they load
try:
    # Load bias server from Bias-Detection/backend/server.py
    bias_path = Path(__file__).resolve().parent.parent / "Bias-Detection" / "backend" / "server.py"
    spec = importlib.util.spec_from_file_location("bias_server_external", bias_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load bias server from {bias_path}")
    bias_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bias_server)

    app.mount("/bias", bias_server.app)
    loaded_bias = True
except Exception as exc:
    print(f"[Combined] Failed to load bias app: {exc}")

try:
    import server.server_claim as claim_server

    app.mount("/claim", claim_server.app)
    loaded_claim = True
except Exception as exc:
    print(f"[Combined] Failed to load claim app: {exc}")


@app.get("/health")
async def health():
    """
    Combined health endpoint that reports the status of both sub-apps.
    """
    return {
        "status": "ok",
        "bias": {
            "loaded": loaded_bias,
            "model_loaded": (
                getattr(bias_server, "model", None) is not None
                if loaded_bias and hasattr(bias_server, "model")
                else getattr(bias_server, "detector", None) is not None
                if loaded_bias and hasattr(bias_server, "detector")
                else False
            ),
        },
        "claim": {
            "loaded": loaded_claim,
            "component": "claim_verification" if loaded_claim else None,
            "model_loaded": getattr(claim_server, "claim_model", None) is not None if loaded_claim else False,
        },
    }
