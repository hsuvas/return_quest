"""
Response parser & validator — converts raw LLM output into typed objects.

Supports two parsing paths:
  A) Native tool calls returned via the provider API.
  B) JSON text in the message content (the prompt instructs JSON output).

Also validates tool names against the known registry.
"""

import json
import re
from typing import Any, Dict, List, Optional

from .conversation_state import Resolution, ToolCallRecord


_COMPOUND_QUESTION_RE = re.compile(
    r'^(.*?(?:let me know|tell me|share(?: with me)?|provide|confirm|clarify))\s+(.+?)(?:,\s*.+)+\?$',
    re.IGNORECASE | re.DOTALL,
)

_FILLER_QUESTION_RE = re.compile(
    r'^(?:'
    r'do you understand'
    r'|does that (?:make sense|sound (?:good|right|okay|ok))'
    r'|(?:is|does) that (?:okay|ok|alright|clear|correct)'
    r'|(?:could|can|would) you (?:please )?confirm(?: the following(?: information)?)?'
    r'|shall (?:we|i) proceed'
    r'|(?:would|do) you (?:like to )?proceed'
    r'|(?:are you )?(?:ready to proceed|ready to continue)'
    r'|(?:is|are) (?:that|there) (?:anything else|everything)'
    r'|(?:do|did) (?:you|that) make sense'
    r')\??\s*$',
    re.IGNORECASE,
)


def _is_filler_question(question: str) -> bool:
    """Return True if question is a hollow procedural filler with no case-specific content."""
    q = question.strip().rstrip("?").strip()
    # Very short questions (≤6 words) matching filler patterns
    if _FILLER_QUESTION_RE.match(question.strip()):
        return True
    # Catch short generic confirmations not covered above (≤5 words, no numbers/proper nouns)
    words = q.split()
    if len(words) <= 5 and not re.search(r'[A-Z][a-z]|\d', q):
        if re.search(r'\b(?:understand|confirm|proceed|okay|ok|sense|sound|clear|alright)\b', q, re.IGNORECASE):
            return True
    return False


def _reduce_compound_question(question: str) -> str:
    """If question_to_customer lists multiple items (comma-joined), keep only the first sub-ask."""
    if not question or question.count(",") < 2:
        return question
    m = _COMPOUND_QUESTION_RE.match(question.strip())
    if m:
        prefix = m.group(1).strip()
        first_item = m.group(3).strip()
        return f"{prefix} {first_item}?"
    return question


def _strip_all_questions(text: str) -> str:
    """Remove every sentence that contains a '?' from text, then clean up orphaned list markers.

    Used to clean the message body when question_to_customer is present,
    ensuring the body carries zero questions.
    """
    if not text or "?" not in text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result = [p for p in parts if "?" not in p]
    joined = " ".join(result).strip()
    # Remove orphaned numbered list markers left after question stripping (e.g. "2. 3.")
    joined = re.sub(r'\b\d+\.\s*', ' ', joined)
    joined = re.sub(r'\s{2,}', ' ', joined).strip()
    return joined


def _enforce_single_question(text: str) -> str:
    """Fallback: remove all question sentences after the first one.

    Used only on the plain-text parse path (no question_to_customer field).
    Splits on sentence-ending punctuation, keeps every non-question sentence
    plus the very first sentence that contains '?', and discards the rest.
    """
    if not text or text.count("?") <= 1:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    result: List[str] = []
    question_seen = False
    for part in parts:
        if "?" in part:
            if not question_seen:
                result.append(part)
                question_seen = True
        else:
            result.append(part)
    return " ".join(result).strip()

# ---------------------------------------------------------------------------
# JSON extraction regexes (mirrors src/dataset/output_collect patterns)
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM text output."""
    t = text.strip()

    # Already valid JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # Fenced code block
    m = _JSON_FENCE_RE.search(t)
    if m:
        return json.loads(m.group(1))

    # Bare object
    m = _JSON_OBJ_RE.search(t)
    if m:
        return json.loads(m.group(1))

    raise ValueError("Could not extract JSON from model output.")


# ---------------------------------------------------------------------------
# Parsed response types
# ---------------------------------------------------------------------------

class AgentResponse:
    """Parsed agent turn — may contain a message, tool calls, or both."""

    def __init__(
        self,
        message: Optional[str] = None,
        tool_calls: Optional[List[ToolCallRecord]] = None,
        facts: Optional[List[str]] = None,
        policy_refs: Optional[List[str]] = None,
        resolution: Optional[Resolution] = None,
        conclusion_reached: bool = False,
        reasoning_summary: Optional[str] = None,
        agent_persona_type: Optional[str] = None,
        raw_json: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.tool_calls = tool_calls
        self.facts = facts or []
        self.policy_refs = policy_refs or []
        self.resolution = resolution
        self.conclusion_reached = conclusion_reached
        self.reasoning_summary = reasoning_summary
        self.agent_persona_type = agent_persona_type
        self.raw_json = raw_json


class CustomerResponse:
    """Parsed customer turn."""

    def __init__(
        self,
        reply: str,
        information_provided: Optional[List[str]] = None,
        emotional_tone: str = "neutral",
        tool_calls: Optional[List[ToolCallRecord]] = None,
        withdraw: bool = False,
    ):
        self.reply = reply
        self.information_provided = information_provided or []
        self.emotional_tone = emotional_tone
        self.tool_calls = tool_calls
        self.withdraw = withdraw


# ---------------------------------------------------------------------------
# Agent response parsing
# ---------------------------------------------------------------------------

def parse_agent_response(llm_response: Any) -> AgentResponse:
    """Parse an ``LLMResponse`` into an ``AgentResponse``.

    Handles both native tool-call responses (Path A) and JSON-in-content
    responses (Path B).
    """
    # Path A: native tool calls from the API
    if llm_response.has_tool_calls:
        tool_calls = [
            ToolCallRecord(
                tool_name=tc["tool_name"],
                tool_call_id=tc["tool_call_id"],
                arguments=tc["arguments"],
            )
            for tc in llm_response.tool_calls
        ]
        # The content may also contain a JSON body with the full agent object
        message = None
        facts: List[str] = []
        policy_refs: List[str] = []
        resolution = None
        conclusion = False
        reasoning = None
        persona = None
        raw = None

        if llm_response.content:
            try:
                raw = extract_json(llm_response.content)
                parsed = _parse_agent_json_body(raw)
                message = parsed.message
                facts = parsed.facts
                policy_refs = parsed.policy_refs
                resolution = parsed.resolution
                conclusion = parsed.conclusion_reached
                reasoning = parsed.reasoning_summary
                persona = parsed.agent_persona_type
            except Exception:
                # Content wasn't valid JSON — use as plain message
                message = _enforce_single_question(llm_response.content.strip()) or None

        return AgentResponse(
            message=message,
            tool_calls=tool_calls,
            facts=facts,
            policy_refs=policy_refs,
            resolution=resolution,
            conclusion_reached=conclusion,
            reasoning_summary=reasoning,
            agent_persona_type=persona,
            raw_json=raw,
        )

    # Path B: JSON text content
    if not llm_response.content:
        raise ValueError("Empty response from agent LLM")

    raw = extract_json(llm_response.content)
    return _parse_agent_json_body(raw)


def _parse_agent_json_body(data: Dict[str, Any]) -> AgentResponse:
    """Parse the JSON body the existing agent prompt instructs the LLM to produce."""
    # Messages from conversation_flow
    messages: List[str] = []
    for item in data.get("conversation_flow", []):
        msg = (item or {}).get("message", "")
        if msg:
            messages.append(msg)

    # Extract the single question from the dedicated field
    question = (data.get("question_to_customer") or "").strip()
    if question and not question.endswith("?"):
        question += "?"
    question = _reduce_compound_question(question)

    body = "\n".join(messages) if messages else ""

    if question and not _is_filler_question(question):
        # Strip any stray questions from the message body — question lives only in question_to_customer
        body = _strip_all_questions(body)
        combined_message = (body.rstrip() + "\n\n" + question).strip() if body else question
    else:
        # Filler/absent question_to_customer — keep real questions from the body
        combined_message = _enforce_single_question(body) if body else None

    # Tool calls from tool_calls_made
    tool_calls: Optional[List[ToolCallRecord]] = None
    raw_tools = data.get("tool_calls_made") or []
    if raw_tools:
        tool_calls = []
        for i, tc in enumerate(raw_tools):
            tool_calls.append(
                ToolCallRecord(
                    tool_name=tc.get("tool_name", "unknown"),
                    tool_call_id=tc.get("tool_call_id", f"call_{i}"),
                    arguments=tc.get("arguments", {}),
                )
            )

    # Resolution
    resolution = None
    conclusion_str = data.get("conclusion_reached", "No")
    conclusion = conclusion_str == "Yes"
    raw_res = data.get("final_resolution")
    if raw_res and conclusion:
        try:
            resolution = Resolution(**raw_res)
        except Exception:
            resolution = None

    return AgentResponse(
        message=combined_message,
        tool_calls=tool_calls,
        facts=data.get("facts_collected_or_assumed", []),
        policy_refs=data.get("policy_references_used", []),
        resolution=resolution,
        conclusion_reached=conclusion,
        reasoning_summary=data.get("reasoning_summary"),
        agent_persona_type=data.get("agent_persona_type"),
        raw_json=data,
    )


# ---------------------------------------------------------------------------
# Customer response parsing
# ---------------------------------------------------------------------------

def parse_customer_response(llm_response: Any) -> CustomerResponse:
    """Parse an ``LLMResponse`` into a ``CustomerResponse``."""
    if not llm_response.content:
        raise ValueError("Empty response from customer LLM")

    data = extract_json(llm_response.content)

    # Optional tool calls
    tool_calls: Optional[List[ToolCallRecord]] = None
    raw_tools = data.get("tool_calls_made") or []
    if raw_tools:
        tool_calls = [
            ToolCallRecord(
                tool_name=tc.get("tool_name", ""),
                tool_call_id=tc.get("tool_call_id", f"cust_call_{i}"),
                arguments=tc.get("arguments", {}),
            )
            for i, tc in enumerate(raw_tools)
        ]

    return CustomerResponse(
        reply=data.get("customer_reply", ""),
        information_provided=data.get("information_provided", []),
        emotional_tone=data.get("emotional_tone", "neutral"),
        tool_calls=tool_calls,
        withdraw=data.get("withdraw_conversation", False),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_tool_call(
    tool_call: ToolCallRecord,
    valid_tool_names: List[str],
) -> bool:
    """Return True if the tool call references a known tool name."""
    return tool_call.tool_name in valid_tool_names
