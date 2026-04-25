import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from env.support_env import SupportEnv
import uvicorn

app = create_app(
    SupportEnv, CallToolAction, CallToolObservation, env_name="support_env"
)
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "index.html"

@app.get("/")
def root():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return JSONResponse(
        {"status": "ok", "service": "openenv-support-agent", "note": "templates/index.html not found"}
    )

@app.get("/healthz")
def healthz():
    return {"ok": True}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()