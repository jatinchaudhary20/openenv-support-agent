import sys
import os

sys.path.append(os.getcwd())

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from env.support_env import SupportEnv
import uvicorn

app = create_app(
    SupportEnv, CallToolAction, CallToolObservation, env_name="support_env"
)

@app.get("/")
def root():
    return {"status": "ok", "service": "openenv-support-agent"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()