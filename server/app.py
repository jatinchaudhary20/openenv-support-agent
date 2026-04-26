from pathlib import Path

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from fastapi.responses import HTMLResponse

from env.support_env import SupportEnv
import uvicorn

app = create_app(
    SupportEnv, CallToolAction, CallToolObservation, env_name="support_env"
)

@app.get("/", response_class=HTMLResponse)
def root():
    template_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    with template_path.open("r", encoding="utf-8") as fp:
        return HTMLResponse(content=fp.read())


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "openenv-support-agent"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()