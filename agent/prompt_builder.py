"""
Prompt builder — constructs messages for the agent and customer LLMs.

Imports the existing prompt templates verbatim from
``src/dataset/output_gen_prompt_agent_multitype.py`` and
``src/dataset/output_gen_prompt_customer_multitype.py``, then fills in the
runtime placeholders using the same ``make_safe_format_string`` technique
used by the dataset pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation_state import ConversationState, ConversationTurn

# ---------------------------------------------------------------------------
# Ensure the dataset package is importable
# ---------------------------------------------------------------------------

# In the standalone return_quest folder, prompt modules live alongside this file
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from output_gen_prompt_agent_multitype import (  # noqa: E402
    output_creation_prompt as _AGENT_PROMPT_RAW,
)
from output_gen_prompt_customer_multitype import (  # noqa: E402
    customer_response_prompt as _CUSTOMER_PROMPT_RAW,
)
from prompt_single import (  # noqa: E402
    single_agent_prompt as _SINGLE_AGENT_PROMPT_RAW,
    single_customer_prompt as _SINGLE_CUSTOMER_PROMPT_RAW,
)

from tool_registry import format_tools_for_prompt_detailed  # noqa: E402


# ---------------------------------------------------------------------------
# Safe format-string construction
# ---------------------------------------------------------------------------

def _make_safe_format_string(raw_prompt: str, allowed_keys: List[str]) -> str:
    """Escape all ``{`` / ``}`` then restore only *allowed_keys*.

    This is required because the prompt templates contain literal JSON
    curly braces in their output-format examples.
    """
    s = raw_prompt.replace("{", "{{").replace("}", "}}")
    for k in allowed_keys:
        s = s.replace("{{" + k + "}}", "{" + k + "}")
    return s


_AGENT_ALLOWED_KEYS = [
    "agent_persona",
    "conversation_type",
    "conversation_variant_id",
    "prior_variants_brief",
    "primary_policy_text",
    "related_policies_text",
    "policy_ambiguities",
    "persona_details",
    "return_scenario_details",
    "conversation_history",
]

_CUSTOMER_ALLOWED_KEYS = [
    "conversation_type",
    "conversation_variant_id",
    "prior_variants_brief",
    "persona_details",
    "return_scenario_details",
    "primary_policy_text",
    "conversation_history",
    "latest_agent_message",
]

_AGENT_PROMPT = _make_safe_format_string(_AGENT_PROMPT_RAW, _AGENT_ALLOWED_KEYS)
_CUSTOMER_PROMPT = _make_safe_format_string(_CUSTOMER_PROMPT_RAW, _CUSTOMER_ALLOWED_KEYS)

# Single-conversation mode keys (no variant_id / prior_variants_brief)
_SINGLE_AGENT_ALLOWED_KEYS = [
    "agent_persona",
    "conversation_type",
    "primary_policy_text",
    "related_policies_text",
    "policy_ambiguities",
    "persona_details",
    "return_scenario_details",
    "conversation_history",
]

_SINGLE_CUSTOMER_ALLOWED_KEYS = [
    "conversation_type",
    "persona_details",
    "return_scenario_details",
    "primary_policy_text",
    "conversation_history",
    "latest_agent_message",
]

_SINGLE_AGENT_PROMPT = _make_safe_format_string(
    _SINGLE_AGENT_PROMPT_RAW, _SINGLE_AGENT_ALLOWED_KEYS
)
_SINGLE_CUSTOMER_PROMPT = _make_safe_format_string(
    _SINGLE_CUSTOMER_PROMPT_RAW, _SINGLE_CUSTOMER_ALLOWED_KEYS
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _related_policies_to_text(related_policies: List[Dict[str, Any]]) -> str:
    if not related_policies:
        return "(none)"
    blocks = []
    for i, rp in enumerate(related_policies, start=1):
        url = rp.get("url", "(missing url)")
        txt = rp.get("text", "")
        blocks.append(f"[Related Policy {i}] {url}\n{txt}".strip())
    return "\n\n".join(blocks).strip()


def _latest_agent_message(history: List[ConversationTurn]) -> Optional[str]:
    for turn in reversed(history):
        if turn.turn == "agent" and turn.message and turn.message.strip():
            return turn.message.strip()
    return None


# ---------------------------------------------------------------------------
# Agent prompt construction
# ---------------------------------------------------------------------------

def build_agent_messages(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    use_native_tools: bool = True,
    single_mode: bool = False,
) -> List[Dict[str, str]]:
    """Return ``[system, user]`` messages for the agent LLM call.

    When *use_native_tools* is True the system prompt is compact (tool
    definitions are passed via the API ``tools`` parameter).  When False,
    the full tool descriptions are injected into the system prompt.

    When *single_mode* is True, uses the single-conversation prompt that
    lets the agent pick the most plausible resolution instead of
    targeting a hard-coded outcome per variant.
    """
    system = _build_agent_system_prompt(use_native_tools)
    if single_mode:
        user = _build_single_agent_user_prompt(scenario, state)
    else:
        user = _build_agent_user_prompt(scenario, state, prior_variants_brief)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_agent_system_prompt(use_native_tools: bool) -> str:
    base = (
        "RUNTIME CONSTRAINT (VERY IMPORTANT):\n"
        "- You are in an iterative loop.\n"
        '- If you still need customer info, set conclusion_reached to "No" '
        "and final_resolution to null.\n"
        '- Only when decision-ready, set conclusion_reached to "Yes" '
        "and output final_resolution.\n"
        "- Output MUST be valid JSON only, matching the schema in the prompt.\n"
    )
    if not use_native_tools:
        base += (
            "\n\n"
            + format_tools_for_prompt_detailed()
            + "\n\n"
            "- MANDATORY: You MUST include at least one tool call in tool_calls_made.\n"
            "- Specify which tools you would call and with what arguments.\n"
            "- Tool results will be provided back to you in the conversation history.\n"
        )
    return base


def _build_agent_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str,
) -> str:
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    related_policies = scenario["Policy"].get("Related policies", [])
    related_blob = _related_policies_to_text(related_policies)

    policy_ambiguities = scenario.get("task", {}).get("related_policy_issues", [])
    if not isinstance(policy_ambiguities, list):
        policy_ambiguities = []

    persona_details = scenario.get("persona", {})
    return_scenario_details = {
        "scenario_id": scenario.get("scenario_id"),
        "task": scenario.get("task", {}),
    }

    return _AGENT_PROMPT.format(
        agent_persona=state.agent_persona,
        conversation_type="multitype",
        conversation_variant_id=str(state.variant_id),
        prior_variants_brief=prior_variants_brief,
        primary_policy_text=primary_policy_text,
        related_policies_text=related_blob,
        policy_ambiguities=_safe_json(policy_ambiguities),
        persona_details=_safe_json(persona_details),
        return_scenario_details=_safe_json(return_scenario_details),
        conversation_history=state.get_formatted_history_str(),
    )


def _build_single_agent_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
) -> str:
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    related_policies = scenario["Policy"].get("Related policies", [])
    related_blob = _related_policies_to_text(related_policies)

    policy_ambiguities = scenario.get("task", {}).get("related_policy_issues", [])
    if not isinstance(policy_ambiguities, list):
        policy_ambiguities = []

    persona_details = scenario.get("persona", {})
    return_scenario_details = {
        "scenario_id": scenario.get("scenario_id"),
        "task": scenario.get("task", {}),
    }

    return _SINGLE_AGENT_PROMPT.format(
        agent_persona=state.agent_persona,
        conversation_type="single",
        primary_policy_text=primary_policy_text,
        related_policies_text=related_blob,
        policy_ambiguities=_safe_json(policy_ambiguities),
        persona_details=_safe_json(persona_details),
        return_scenario_details=_safe_json(return_scenario_details),
        conversation_history=state.get_formatted_history_str(),
    )


# ---------------------------------------------------------------------------
# Customer prompt construction
# ---------------------------------------------------------------------------

def build_customer_messages(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str = "(none)",
    single_mode: bool = False,
) -> List[Dict[str, str]]:
    """Return ``[system, user]`` messages for the customer LLM call."""
    system = "Output ONLY valid JSON matching the schema in the prompt."
    if single_mode:
        user = _build_single_customer_user_prompt(scenario, state)
    else:
        user = _build_customer_user_prompt(scenario, state, prior_variants_brief)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_customer_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
    prior_variants_brief: str,
) -> str:
    persona_details = scenario.get("persona", {})
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    return_scenario_details = {
        "scenario_id": scenario.get("scenario_id"),
        "task": scenario.get("task", {}),
    }
    latest = _latest_agent_message(state.history) or "(none yet)"

    return _CUSTOMER_PROMPT.format(
        conversation_type="multitype",
        conversation_variant_id=str(state.variant_id),
        prior_variants_brief=prior_variants_brief,
        persona_details=_safe_json(persona_details),
        return_scenario_details=_safe_json(return_scenario_details),
        primary_policy_text=primary_policy_text,
        conversation_history=state.get_formatted_history_str(),
        latest_agent_message=latest,
    )


def _build_single_customer_user_prompt(
    scenario: Dict[str, Any],
    state: ConversationState,
) -> str:
    persona_details = scenario.get("persona", {})
    primary_policy_text = scenario["Policy"]["Primary Policy"]["text"]
    return_scenario_details = {
        "scenario_id": scenario.get("scenario_id"),
        "task": scenario.get("task", {}),
    }
    latest = _latest_agent_message(state.history) or "(none yet)"

    return _SINGLE_CUSTOMER_PROMPT.format(
        conversation_type="single",
        persona_details=_safe_json(persona_details),
        return_scenario_details=_safe_json(return_scenario_details),
        primary_policy_text=primary_policy_text,
        conversation_history=state.get_formatted_history_str(),
        latest_agent_message=latest,
    )
