"""
FastAPI backend for the Data Scientist AI (CrewAI multi-agent)
==============================================================
Run:  uvicorn main:app --reload

Session-based workspaces: every chat session owns workspace/<session_id>/,
so uploads, parquet checkpoints, plots, report.md, project memory and the
notebook log of one session never leak into another.

  POST /api/session                 create a session -> {"session_id": ...}
  POST /api/upload?session_id=      save dataset -> 5-row preview
  GET  /api/preview?session_id=&filename=
  GET  /api/report?session_id=
  GET  /api/notebook?session_id=
  GET  /api/plots?session_id=
  GET  /api/files/{name}?session_id=
  POST /api/reset?session_id=       clear this session's workspace
  WS   /ws?session_id=              chat: run crews, stream logs, answer questions
"""

import os
import glob
import json
import shutil
import asyncio
import threading

import pandas as pd
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from crew_ds import (
    Session, run_crew, export_notebook, reset_notebook_log,
    new_session_id, session_dir, notebook_log_path, project_memory_path,
)

app = FastAPI(title="Data Scientist AI")

# Bedrock region for every LLM call (override with the AWS_REGION env var).
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")

ALLOWED_EXT = (".csv", ".xlsx", ".xls", ".parquet", ".json")


def _ws_dir(session_id: str) -> str:
    """Resolve (and create) the workspace for a session id."""
    return session_dir(session_id)


def _preview(ws: str, filename: str, n: int = 5):
    path = os.path.join(ws, os.path.basename(filename))
    if filename.endswith(".csv"):
        df = pd.read_csv(path, nrows=n)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, nrows=n)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(path).head(n)
    elif filename.endswith(".json"):
        df = pd.read_json(path).head(n)
    else:
        return None
    df = df.head(n)
    return {"columns": [str(c) for c in df.columns],
            "rows": json.loads(df.astype(str).to_json(orient="values"))}


@app.post("/api/session")
def create_session():
    """The browser calls this once on load and reuses the id for everything."""
    sid = new_session_id()
    session_dir(sid)  # create the directory now
    return {"session_id": sid}


@app.post("/api/upload")
async def upload(session_id: str = Query(...), file: UploadFile = File(...)):
    ws = _ws_dir(session_id)
    name = os.path.basename(file.filename or "")
    if not name.endswith(ALLOWED_EXT):
        return JSONResponse({"error": f"Unsupported type. Use {ALLOWED_EXT}"},
                            status_code=400)
    with open(os.path.join(ws, name), "wb") as f:
        f.write(await file.read())
    try:
        return {"filename": name, "preview": _preview(ws, name)}
    except Exception as e:
        return JSONResponse({"filename": name, "preview": None,
                             "warning": f"Saved, but could not preview: {e}"})


@app.get("/api/preview")
def preview(session_id: str = Query(...), filename: str = Query(...)):
    try:
        return {"filename": filename,
                "preview": _preview(_ws_dir(session_id), filename)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/report")
def report(session_id: str = Query(...)):
    path = os.path.join(_ws_dir(session_id), "report.md")
    if not os.path.exists(path):
        return JSONResponse({"exists": False})
    with open(path, encoding="utf-8") as f:
        return {"exists": True, "markdown": f.read()}


@app.get("/api/notebook")
def notebook(session_id: str = Query(...)):
    ws = _ws_dir(session_id)
    if not os.path.exists(notebook_log_path(ws)):
        return JSONResponse({"error": "No cells have run yet."}, status_code=404)
    return FileResponse(export_notebook(ws), filename="pipeline.ipynb",
                        media_type="application/x-ipynb+json")


@app.get("/api/plots")
def plots(session_id: str = Query(...)):
    ws = _ws_dir(session_id)
    return {"plots": sorted(os.path.basename(p) for p in
                            glob.glob(os.path.join(ws, "*.png")))}


@app.get("/api/files/{name}")
def get_file(name: str, session_id: str = Query(...)):
    name = os.path.basename(name)  # no path traversal
    path = os.path.join(_ws_dir(session_id), name)
    if not os.path.exists(path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path)


@app.post("/api/reset")
def reset(session_id: str = Query(...)):
    """Wipe this session's workspace entirely and hand back a fresh id."""
    ws = _ws_dir(session_id)
    shutil.rmtree(ws, ignore_errors=True)
    sid = new_session_id()
    session_dir(sid)
    return {"ok": True, "session_id": sid}


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------
# Browser -> server:
#   {"type": "start",  "content": "...", "mode": "autopilot"|"copilot",
#    "model": "bedrock/..."}
#   {"type": "answer", "content": "..."}     (reply to a "question" event)
#   {"type": "stop"}                         cancel the running crew
# Server -> browser (mirrors crew_ds.Session.emit):
#   info | pip | stdout | vision | error | question | answer | stage |
#   files_changed | done
@app.websocket("/ws")
async def ws(websocket: WebSocket, session_id: str = Query(None)):
    await websocket.accept()
    session = Session(session_id)          # binds to workspace/<session_id>/
    await websocket.send_json({"type": "info",
                               "content": f"session {session.id} ready"})
    worker: threading.Thread | None = None
    pipeline_done = False   # True once the full autopilot pipeline has run
    loop = asyncio.get_event_loop()

    async def forward_outbox():
        while True:
            msg = await loop.run_in_executor(None, session.outbox.get)
            await websocket.send_json(msg)

    forwarder = asyncio.create_task(forward_outbox())
    try:
        while True:
            data = await websocket.receive_json()
            kind = data.get("type")
            content = data.get("content", "")

            if kind == "answer":
                session.answers.put(content)

            elif kind == "stop":
                if worker and worker.is_alive():
                    session.cancelled = True
                    session.answers.put("")   # unblock a pending ask_user
                    await websocket.send_json(
                        {"type": "info",
                         "content": "Stopping after the current step…"})
                else:
                    await websocket.send_json(
                        {"type": "info", "content": "Nothing is running."})

            elif kind == "start":
                if worker and worker.is_alive():
                    await websocket.send_json(
                        {"type": "error",
                         "content": "A run is already in progress. If the crew "
                                    "asked a question, reply to it instead."})
                    continue

                model = data.get("model") or "bedrock/global.amazon.nova-2-lite-v1:0"
                try:
                    from crewai import LLM

                    llm = LLM(model=model, region_name=AWS_REGION)
                except Exception as e:
                    await websocket.send_json({"type": "error",
                                               "content": f"LLM config: {e}"})
                    continue

                # First autopilot run = the full pipeline. Every later message
                # is a follow-up: build on the existing artifacts instead of
                # re-interviewing and redoing EDA from scratch.
                session.cancelled = False   # fresh run, clear any old stop
                requested = data.get("mode", "autopilot")
                mode = ("followup" if requested == "autopilot" and pipeline_done
                        else requested)

                worker = threading.Thread(
                    target=run_crew, args=(mode, llm, session, content),
                    daemon=True)
                worker.start()
                if mode == "autopilot":
                    pipeline_done = True
    except WebSocketDisconnect:
        pass
    finally:
        forwarder.cancel()
        session.answers.put("")  # unblock a crew thread stuck on ask_user


# Serve the frontend last so /api/* and /ws win the routing
app.mount("/", StaticFiles(directory="static", html=True), name="static")