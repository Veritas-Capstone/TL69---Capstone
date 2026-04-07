"""
Entry point that exposes both the bias-detection API and claim-verification API
under separate prefixes on a single FastAPI app.

Run from repo root:
    uvicorn server.server_combined:app --reload --port 8000

Routes:
    /bias/...   -> existing bias detection endpoints (e.g., /bias/analyze, /bias/health)
    /claim/...  -> existing claim verification endpoints (e.g., /claim/verify-claims-from-passage)
"""

import os
import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _configure_windows_java() -> None:
    if os.name != "nt":
        return

    current_java_home = os.environ.get("JAVA_HOME", "")
    current_jvm_path = os.environ.get("JVM_PATH", "")
    if current_jvm_path and Path(current_jvm_path).exists():
        return

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    adoptium_root = Path(program_files) / "Eclipse Adoptium"
    if not adoptium_root.exists():
        return

    candidates = sorted(adoptium_root.glob("jdk-21*"))
    for candidate in candidates:
        jvm_path = candidate / "bin" / "server" / "jvm.dll"
        if not jvm_path.exists():
            continue

        os.environ["JAVA_HOME"] = str(candidate)
        os.environ["JVM_PATH"] = str(jvm_path)

        java_bin = candidate / "bin"
        path_value = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{java_bin};{path_value}"

        if current_java_home and current_java_home != str(candidate):
            print(f"[Combined] JAVA_HOME updated from {current_java_home} to {candidate}")
        print(f"[Combined] Using Java 21 for pyjnius: {jvm_path}")
        return


_configure_windows_java()

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


def _load_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Mount existing apps under prefixes if they load
try:
    repo_root = Path(__file__).resolve().parents[1]
    bias_backend_path = repo_root / "Bias-Detection" / "backend" / "server.py"
    bias_server = _load_module_from_file("bias_backend_server", bias_backend_path)

    app.mount("/bias", bias_server.app)
    loaded_bias = True
except Exception as exc:
    print(f"[Combined] Failed to load backend bias app: {exc}")
    try:
        import server.server as bias_server

        app.mount("/bias", bias_server.app)
        loaded_bias = True
        print("[Combined] Falling back to server.server for /bias routes.")
    except Exception as fallback_exc:
        print(f"[Combined] Failed fallback bias app load: {fallback_exc}")

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
            "model_loaded": getattr(bias_server, "model", None) is not None if loaded_bias else False,
        },
        "claim": {
            "loaded": loaded_claim,
            "component": "claim_verification" if loaded_claim else None,
            "model_loaded": getattr(claim_server, "claim_model", None) is not None if loaded_claim else False,
        },
    }
