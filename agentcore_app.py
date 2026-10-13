"""
AgentCore Runtime entrypoint for the CrewAI Data Scientist
==========================================================
This replaces the FastAPI/WebSocket server (main.py) with an Amazon Bedrock
AgentCore Runtime app. The CrewAI logic in crew_ds.py is reused unchanged.

AgentCore Runtime contract
--------------------------
  * the container listens on 0.0.0.0:8080 and exposes POST /invocations + GET
    /ping — BedrockAgentCoreApp does this for us.
  * one @app.entrypoint handler receives (payload, context); `context.session_id`
    is the runtime session id supplied by the caller.
  * `yield`ing from an async handler streams events back to the caller as SSE.
  * the SAME session id reuses the SAME warm microVM across invocations for up
    to 8h. We exploit this for human-in-the-loop: a crew thread that blocks on
    ask_user in one invocation is resumed by the next invocation's answer.

Payload actions
---------------
  start :  {"action":"start", "prompt":"…",
            "mode":"autopilot"|"copilot",              # default autopilot
            "model":"bedrock/…",                       # optional
            "dataset_s3":"s3://bucket/key.csv",        # optional dataset in
            "dataset_b64":"…","filename":"data.csv"}   #   (or inline, small)
  answer:  {"action":"answer", "content":"…"}          # reply to a "question"
  stop  :  {"action":"stop"}                           # cancel the running crew

Streamed events (same shapes as the old WebSocket messages)
  info | pip | stdout | vision | error | question | answer | stage |
  files_changed | artifacts | done
A "question" event PAUSES the run (the crew thread stays alive, blocked). Send
an {"action":"answer"} invocation on the SAME session id to resume. A "done"
event ends the run; it is preceded by an "artifacts" event carrying the
presigned S3 URLs of everything the pipeline produced.

Local testing:  python agentcore_app.py   ->   serves on http://0.0.0.0:8080
  curl -N -XPOST localhost:8080/invocations -d '{"action":"start","prompt":"…"}'
"""

import os
import queue
import base64
import asyncio
import threading

from bedrock_agentcore import BedrockAgentCoreApp

# ---------------------------------------------------------------------------
# Config (all overridable via the AgentCore Runtime environment)
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET", "")
ARTIFACT_PREFIX = os.environ.get("ARTIFACT_PREFIX", "ds-agent")
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL", "bedrock/global.amazon.nova-2-lite-v1:0")
# Poll granularity when draining the crew's event queue (seconds).
_POLL = 1.0

# AgentCore microVMs are Linux; /tmp is the only reliably writable location.
# Guard against a WORKSPACE_ROOT that a Windows/MSYS shell mangled at launch
# time — e.g. "/tmp/workspace" rewritten to "C:/Users/.../workspace", which on
# Linux makes os.path.abspath() yield an unwritable "/app/C:/..." path. This
# runs before crew_ds is imported (that import is lazy, inside the handler),
# so the corrected value is what crew_ds reads.
_ws = os.environ.get("WORKSPACE_ROOT", "")
if (not _ws) or (":" in _ws) or ("\\" in _ws) or (not _ws.startswith("/")):
    os.environ["WORKSPACE_ROOT"] = "/tmp/workspace"

app = BedrockAgentCoreApp()


# ---------------------------------------------------------------------------
# Per-session run registry
# ---------------------------------------------------------------------------
# One entry per active runtime session, keyed by context.session_id. Because
# the microVM stays warm across invocations for a session, a paused crew thread
# (blocked in ask_user) survives here between the "start" and "answer" calls.
class _Run:
    __slots__ = ("session", "worker")

    def __init__(self, session, worker):
        self.session = session
        self.worker = worker


_RUNS: dict[str, _Run] = {}
_RUNS_LOCK = threading.Lock()


def _get(sid: str):
    with _RUNS_LOCK:
        return _RUNS.get(sid)


def _put(sid: str, run: _Run):
    with _RUNS_LOCK:
        _RUNS[sid] = run


def _drop(sid: str):
    with _RUNS_LOCK:
        _RUNS.pop(sid, None)


def _session_id(context, payload: dict) -> str:
    """Prefer the runtime-supplied session id; fall back to the payload."""
    sid = getattr(context, "session_id", None) if context is not None else None
    return sid or payload.get("session_id") or "default"


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
async def _finish(run: _Run, sid: str, loop):
    """Export the notebook, push all artifacts to S3, close the run."""
    ws = run.session.workspace
    try:
        from crew_ds import export_notebook, notebook_log_path
        if os.path.exists(notebook_log_path(ws)):
            await loop.run_in_executor(None, export_notebook, ws)
    except Exception as e:                     # notebook export is best-effort
        yield {"type": "info", "content": f"notebook export skipped: {e}"}

    if ARTIFACT_BUCKET:
        try:
            import s3_store
            prefix = f"{ARTIFACT_PREFIX}/{sid}"
            urls = await loop.run_in_executor(
                None, s3_store.upload_dir_presigned, ws, ARTIFACT_BUCKET, prefix)
            yield {"type": "artifacts", "content": urls}
        except Exception as e:                 # never let S3 hide the result
            yield {"type": "info", "content": f"artifact upload failed: {e}"}
    else:
        yield {"type": "info",
               "content": "ARTIFACT_BUCKET not set — artifacts kept only in the "
                          "session workspace and not uploaded."}

    _drop(sid)
    yield {"type": "done", "content": ""}


async def _drain(run: _Run, sid: str, loop):
    """Yield the crew's events until it asks a question (pause) or finishes."""
    outbox = run.session.outbox
    while True:
        try:
            msg = await loop.run_in_executor(None, outbox.get, True, _POLL)
        except queue.Empty:
            if not run.worker.is_alive():
                # Thread ended without a "done" (shouldn't happen — run_crew
                # emits it in finally — but never hang the caller).
                async for ev in _finish(run, sid, loop):
                    yield ev
                return
            continue

        if msg.get("type") == "done":
            # Replace the crew's bare "done" with artifact upload + our done.
            async for ev in _finish(run, sid, loop):
                yield ev
            return

        yield msg
        if msg.get("type") == "question":
            return                              # paused; thread stays blocked


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
@app.entrypoint
async def invoke(payload, context=None):
    loop = asyncio.get_event_loop()
    payload = payload or {}
    sid = _session_id(context, payload)
    action = (payload.get("action") or "start").lower()

    # -- stop ---------------------------------------------------------------
    if action == "stop":
        run = _get(sid)
        if run and run.worker.is_alive():
            run.session.cancelled = True
            run.session.answers.put("")          # unblock a pending ask_user
            yield {"type": "info",
                   "content": "Stopping after the current step…"}
        else:
            yield {"type": "info", "content": "Nothing is running."}
        return

    # -- presign_upload (dataset in, without browser AWS creds) ------------
    if action == "presign_upload":
        if not ARTIFACT_BUCKET:
            yield {"type": "error", "content": "ARTIFACT_BUCKET not configured."}
            return
        import s3_store
        fn = os.path.basename(payload.get("filename") or "dataset.csv")
        key = f"{ARTIFACT_PREFIX}/{sid}/uploads/{fn}"
        try:
            put_url = await loop.run_in_executor(
                None, s3_store.presign_put, ARTIFACT_BUCKET, key)
            yield {"type": "upload_url",
                   "content": {"put_url": put_url,
                               "s3_uri": f"s3://{ARTIFACT_BUCKET}/{key}",
                               "filename": fn}}
        except Exception as e:
            yield {"type": "error", "content": f"presign failed: {e}"}
        return

    # -- answer (resume a paused run) --------------------------------------
    if action == "answer":
        run = _get(sid)
        if not run or not run.worker.is_alive():
            yield {"type": "error",
                   "content": "No paused run for this session (it may have timed "
                              "out). Start a new run with action='start'."}
            return
        run.session.answers.put(payload.get("content", ""))
        async for ev in _drain(run, sid, loop):
            yield ev
        return

    # -- start --------------------------------------------------------------
    existing = _get(sid)
    if existing and existing.worker.is_alive():
        yield {"type": "error",
               "content": "A run is already in progress on this session. If it "
                          "asked a question, reply with action='answer' instead."}
        return

    # Heavy imports are done lazily: CrewAI takes >30s to import and would blow
    # the AgentCore init timeout if imported at module load.
    from crewai import LLM
    from crew_ds import Session, run_crew

    session = Session(sid)
    yield {"type": "info", "content": f"session {session.id} ready"}

    # Dataset in: from S3 (preferred) or inline base64 (small files only).
    dataset_s3 = payload.get("dataset_s3")
    if dataset_s3:
        try:
            import s3_store
            name = await loop.run_in_executor(
                None, s3_store.download, dataset_s3, session.workspace)
            yield {"type": "info", "content": f"loaded {name} from {dataset_s3}"}
        except Exception as e:
            yield {"type": "error",
                   "content": f"could not fetch dataset {dataset_s3}: {e}"}
            yield {"type": "done", "content": ""}
            return
    elif payload.get("dataset_b64"):
        fn = os.path.basename(payload.get("filename") or "dataset.csv")
        try:
            with open(os.path.join(session.workspace, fn), "wb") as f:
                f.write(base64.b64decode(payload["dataset_b64"]))
            yield {"type": "info", "content": f"loaded {fn} from inline payload"}
        except Exception as e:
            yield {"type": "error", "content": f"bad dataset_b64: {e}"}
            yield {"type": "done", "content": ""}
            return

    model = payload.get("model") or DEFAULT_MODEL
    try:
        llm = LLM(model=model, region_name=AWS_REGION)
    except Exception as e:
        yield {"type": "error", "content": f"LLM config: {e}"}
        yield {"type": "done", "content": ""}
        return

    mode = payload.get("mode", "autopilot")
    session.cancelled = False
    worker = threading.Thread(
        target=run_crew, args=(mode, llm, session, payload.get("prompt", "")),
        daemon=True)
    worker.start()
    run = _Run(session, worker)
    _put(sid, run)

    async for ev in _drain(run, sid, loop):
        yield ev


if __name__ == "__main__":
    # Serves POST /invocations + GET /ping on 0.0.0.0:8080.
    app.run()
