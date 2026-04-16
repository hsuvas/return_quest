"""
server.py — FastAPI backend for Return Quest showcase.

Manages conversation sessions server-side so the Streamlit frontend only
needs to send (session_id, new_message) per turn. Agent turn and hint
generation run in parallel via ThreadPoolExecutor.

Start automatically from app.py (daemon thread) or manually:
    uvicorn showcase_lnr.server:app --host 127.0.0.1 --port 8765
"""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_SHOWCASE_DIR = Path(__file__).resolve().parent
_AGENT_DIR = str(_SHOWCASE_DIR / "agent")

for _d in (_AGENT_DIR, str(_SHOWCASE_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from showcase_backend import (  # noqa: E402
    run_agent_turn,
    suggest_next_message,
    make_provider,
    make_conversation_state,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Return Quest API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# Sessions hold server-side state that is too large / not serializable to
# pass over the wire on every request.
# ---------------------------------------------------------------------------

_sessions: Dict[str, Dict[str, Any]] = {}

HINT_MODEL = "gpt-4o-mini"
AGENT_TIMEOUT = 90   # seconds
HINT_TIMEOUT  = 30   # seconds


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    scenario: Dict[str, Any]
    agent_persona: str
    model: str
    first_message: str


class TurnRequest(BaseModel):
    message: str


class RejectRequest(BaseModel):
    message: str = "I'm not satisfied with this resolution and would like to discuss further options."


class TurnResponse(BaseModel):
    agent_message: str
    tool_calls: list
    finished: bool
    resolution: Optional[Dict[str, Any]]
    hint: Optional[str]
    tool_calls_count: int
    reasoning: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolution_to_dict(resolution) -> Optional[Dict[str, Any]]:
    """Serialize a Resolution object to a plain dict for JSON transport."""
    if resolution is None:
        return None
    return {
        "resolution_type": getattr(resolution, "resolution_type", ""),
        "resolution_description": getattr(resolution, "resolution_description", ""),
        "conditions": getattr(resolution, "conditions", []),
        "customer_next_steps": getattr(resolution, "customer_next_steps", ""),
    }


def _run_turn(session_id: str) -> TurnResponse:
    """Execute one agent turn. Hint is generated on explicit request only."""
    sess = _sessions[session_id]
    conv_state = sess["conv_state"]
    scenario   = sess["scenario"]
    provider   = sess["provider"]
    agent_persona = sess["agent_persona"]

    try:
        message, tool_results, agent_resp = run_agent_turn(
            conv_state, scenario, provider, agent_persona,
        )
    except Exception as exc:
        import traceback
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}\n{traceback.format_exc()}")

    return TurnResponse(
        agent_message=message,
        tool_calls=tool_results,
        finished=conv_state.finished,
        resolution=_resolution_to_dict(conv_state.resolution),
        hint=None,
        tool_calls_count=len(conv_state.tool_interactions),
        reasoning=getattr(agent_resp, "reasoning_summary", None),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/session")
def create_session(req: CreateSessionRequest):
    """Create a session, append first customer message, run first agent turn."""
    session_id = str(uuid.uuid4())

    provider = make_provider(
        model=req.model,
        temperature=0.7,
        max_tokens=2500,
    )
    conv_state = make_conversation_state(
        scenario=req.scenario,
        agent_persona=req.agent_persona,
    )
    conv_state.append_customer_message(req.first_message)

    _sessions[session_id] = {
        "conv_state": conv_state,
        "scenario": req.scenario,
        "provider": provider,
        "agent_persona": req.agent_persona,
    }

    turn_resp = _run_turn(session_id)
    return {"session_id": session_id, **turn_resp.model_dump()}


@app.post("/api/session/{session_id}/turn")
def send_turn(session_id: str, req: TurnRequest):
    """Append a customer message and run the next agent turn."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = _sessions[session_id]
    sess["conv_state"].append_customer_message(req.message)

    turn_resp = _run_turn(session_id)
    return turn_resp.model_dump()


@app.post("/api/session/{session_id}/reject")
def reject_resolution(session_id: str, req: RejectRequest):
    """Reset finished state, append nudge message, run next agent turn."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    sess = _sessions[session_id]
    conv_state = sess["conv_state"]
    conv_state.finished = False
    conv_state.resolution = None
    conv_state.append_customer_message(req.message)

    turn_resp = _run_turn(session_id)
    return turn_resp.model_dump()


@app.get("/api/session/{session_id}/hint")
def get_hint(session_id: str):
    """Generate a fresh hint for the current conversation state."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sess = _sessions[session_id]
    hint_provider = make_provider(model=HINT_MODEL, temperature=0.7, max_tokens=200)
    try:
        hint = suggest_next_message(
            sess["conv_state"], sess["scenario"], hint_provider
        )
    except Exception as exc:
        hint = f"(Hint error: {exc})"
    return {"hint": hint}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Clean up a session."""
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}
