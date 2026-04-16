"""
Orchestrator — owns the turn loop and routes messages between
Agent, Customer, and Environment.

Also provides ``run_multitype_conversations`` which generates
multiple conversation variants per scenario (with deduplication)
and flattens the output into the JSONL format expected by
``src/evaluation/evaluator.py``.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent import AgentInterface, LLMAgent, LLMCustomer
from conversation_state import ConversationState, Resolution, ToolCallRecord
from environment import Environment
from llm_provider import LLMProvider
from response_parser import validate_tool_call
from tool_registry import get_agent_tools, get_customer_tools, get_tool_names


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_PERSONAS = ["DIRECT", "FAIR", "AGREEABLE", "HELPFUL", "VERY_HELPFUL"]


# ---------------------------------------------------------------------------
# Jaccard deduplication (mirrors src/dataset/ implementation)
# ---------------------------------------------------------------------------

def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set(s: str) -> set:
    s = _normalize_text(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return {t for t in s.split() if t}


def _jaccard(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _conversation_signature(history: List[Dict[str, Any]]) -> str:
    parts = [t.get("message", "") for t in history if t.get("turn") == "agent"]
    return _normalize_text("\n".join(parts))


def _resolution_signature(res: Optional[Dict[str, Any]]) -> str:
    if not res:
        return ""
    fields = [
        res.get("resolution_id", ""),
        res.get("resolution_description", ""),
        " ".join(res.get("conditions", []) or []),
        res.get("customer_next_steps", ""),
    ]
    return _normalize_text("\n".join(fields))


def _is_too_similar(
    new_rec: Dict[str, Any],
    prior_records: List[Dict[str, Any]],
    conv_threshold: float = 0.72,
    res_threshold: float = 0.78,
) -> bool:
    new_conv = new_rec.get("conversation_history", [])
    new_agent_obj = new_rec.get("agent_final_object") or {}
    new_res = new_agent_obj.get("final_resolution") if isinstance(new_agent_obj, dict) else None

    new_conv_sig = _conversation_signature(new_conv)
    new_res_sig = _resolution_signature(new_res)

    for prior in prior_records:
        pc = prior.get("conversation_history", [])
        pa = prior.get("agent_final_object") or {}
        pr = pa.get("final_resolution") if isinstance(pa, dict) else None

        conv_sim = _jaccard(new_conv_sig, _conversation_signature(pc))
        res_sim = _jaccard(new_res_sig, _resolution_signature(pr))

        if (conv_sim >= conv_threshold and res_sim >= res_threshold) or res_sim >= 0.88:
            return True
    return False


def _build_prior_variants_brief(variant_records: List[Dict[str, Any]]) -> str:
    if not variant_records:
        return "(none)"
    lines = []
    for v in variant_records:
        vid = v.get("conversation_variant_id")
        agent_obj = v.get("agent_final_object") or {}
        res = agent_obj.get("final_resolution") if isinstance(agent_obj, dict) else None
        res_desc = (res or {}).get("resolution_description", "") if isinstance(res, dict) else ""
        res_steps = (res or {}).get("customer_next_steps", "") if isinstance(res, dict) else ""
        lines.append(
            f"- Variant {vid}: resolution={_normalize_text(res_desc)[:160]} "
            f"| next_steps={_normalize_text(res_steps)[:160]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator — single conversation variant
# ---------------------------------------------------------------------------

class Orchestrator:
    """Runs the Agent ⇄ Customer ⇄ Environment turn loop for one variant."""

    def __init__(
        self,
        agent: AgentInterface,
        customer: LLMCustomer,
        environment: Environment,
        scenario: Dict[str, Any],
        variant_id: int,
        agent_persona: str,
        max_turns: int = 10,
        max_errors: int = 3,
        sleep_between_turns: float = 0.0,
    ):
        self.agent = agent
        self.customer = customer
        self.environment = environment
        self.scenario = scenario
        self.variant_id = variant_id
        self.agent_persona = agent_persona
        self.max_turns = max_turns
        self.max_errors = max_errors
        self.sleep_between_turns = sleep_between_turns

        self.state = ConversationState(
            scenario=scenario,
            variant_id=variant_id,
            agent_persona=agent_persona,
        )

        self._valid_agent_tools = get_tool_names(get_agent_tools())
        self._valid_customer_tools = get_tool_names(get_customer_tools())

    # ----- main loop -----

    def run(self, prior_variants_brief: str = "(none)") -> Dict[str, Any]:
        """Execute the full conversation and return a per-variant record."""

        # 1. Seed history with the customer's opening message
        opening = self._seed_customer_opening()
        if opening:
            self.state.append_customer_message(opening)

        for turn_i in range(self.max_turns + 1):
            # ---- AGENT TURN ----
            try:
                agent_resp = self.agent.generate_response(
                    state=self.state,
                    prior_variants_brief=prior_variants_brief,
                )
            except Exception as e:
                print(f"  [Orchestrator] Agent error (turn {turn_i}): {e}")
                self.state.error_count += 1
                if self.state.error_count >= self.max_errors:
                    break
                continue

            # Process tool calls → route through environment
            if agent_resp.tool_calls:
                for tc in agent_resp.tool_calls:
                    if not validate_tool_call(tc, self._valid_agent_tools):
                        self.state.error_count += 1
                        continue
                    was_new = self.state.append_tool_call(tc, caller="agent")
                    if was_new:
                        tool_result = self.environment.execute_tool(tc)
                        self.state.append_tool_result(tool_result)

            # Process agent message
            if agent_resp.message:
                self.state.append_agent_message(agent_resp.message)

            # Update metadata from agent response
            if agent_resp.facts:
                self.state.agent_facts = agent_resp.facts
            if agent_resp.reasoning_summary:
                self.state.agent_summary = agent_resp.reasoning_summary
            if agent_resp.agent_persona_type:
                self.state.agent_persona_type = agent_resp.agent_persona_type

            # Check conclusion
            if agent_resp.conclusion_reached and agent_resp.resolution:
                self.state.resolution = agent_resp.resolution
                # Execute process_return with the determined resolution type
                self._execute_process_return(agent_resp.resolution)
                self.state.finished = True
                break

            if turn_i >= self.max_turns:
                break

            if self.sleep_between_turns > 0:
                time.sleep(self.sleep_between_turns)

            # ---- CUSTOMER TURN ----
            try:
                cust_resp = self.customer.generate_response(
                    state=self.state,
                    prior_variants_brief=prior_variants_brief,
                )
            except Exception as e:
                print(f"  [Orchestrator] Customer error (turn {turn_i}): {e}")
                self.state.error_count += 1
                if self.state.error_count >= self.max_errors:
                    break
                continue

            # Process customer tool calls
            if cust_resp.tool_calls:
                for tc in cust_resp.tool_calls:
                    if validate_tool_call(tc, self._valid_customer_tools):
                        was_new = self.state.append_tool_call(tc, caller="customer")
                        if was_new:
                            tool_result = self.environment.execute_tool(tc)
                            self.state.append_tool_result(tool_result)

            # Append customer message
            if cust_resp.reply:
                self.state.append_customer_message(cust_resp.reply)

            # Check customer withdrawal
            if cust_resp.withdraw:
                self.state.customer_withdrew = True
                self.state.finished = True
                break

            if self.sleep_between_turns > 0:
                time.sleep(self.sleep_between_turns)

        return self._build_variant_record()

    # ----- helpers -----

    def _execute_process_return(self, resolution: Resolution) -> None:
        """Call process_return through the environment with the resolution type."""
        task = self.scenario.get("task", {})
        order_id = (
            task.get("order_id")
            or task.get("order_number")
            or self.scenario.get("scenario_id", "UNKNOWN")
        )
        customer_id = (
            self.scenario.get("persona", {}).get("customer_id")
            or self.scenario.get("persona", {}).get("Name", "UNKNOWN")
        )

        tc = ToolCallRecord(
            tool_name="process_return",
            tool_call_id=f"process_return_{self.variant_id}",
            arguments={
                "order_id": str(order_id),
                "customer_id": str(customer_id),
                "resolution_type": resolution.resolution_type,
                "items_to_return": [],
                "return_reason": "other",
                "return_reason_details": resolution.resolution_description,
            },
        )
        was_new = self.state.append_tool_call(tc, caller="agent")
        if was_new:
            tool_result = self.environment.execute_tool(tc)
            self.state.append_tool_result(tool_result)

    def _seed_customer_opening(self) -> str:
        opening = self.scenario.get("first_customer_message", "")
        if opening and isinstance(opening, str):
            return opening
        return self.scenario.get("task", {}).get("task", "")

    def _build_variant_record(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario.get("scenario_id", "unknown"),
            "conversation_type": "multitype",
            "conversation_variant_id": self.variant_id,
            "agent_persona": self.state.agent_persona_type or self.agent_persona,
            "finished": self.state.finished,
            "customer_withdrew": self.state.customer_withdrew,
            "conversation_history": self.state.get_history_dicts(),
            "tool_interactions": self.state.tool_interactions,
            "customer_tool_interactions": self.state.customer_tool_interactions,
            "agent_final_object": {
                "facts_collected_or_assumed": self.state.agent_facts,
                "reasoning_summary": self.state.agent_summary,
                "final_resolution": (
                    self.state.resolution.model_dump() if self.state.resolution else None
                ),
            },
            "source_scenario": self.scenario,
        }


# ---------------------------------------------------------------------------
# Multi-variant runner
# ---------------------------------------------------------------------------

def run_multitype_conversations(
    scenario: Dict[str, Any],
    llm_provider: LLMProvider,
    agent_persona: str = "FAIR",
    num_variants: int = 5,
    max_turns: int = 10,
    max_errors: int = 3,
    use_native_tools: bool = True,
    sleep_between_turns: float = 0.0,
    max_variant_attempts: int = 4,
    include_resolution: bool = True,
) -> Dict[str, Any]:
    """Run conversation variants for one scenario.

    When *num_variants* is 1, runs in single-conversation mode: one
    conversation where the agent picks the most plausible resolution,
    no deduplication, and the single-conversation prompts are used.

    When *num_variants* > 1, applies Jaccard-based deduplication between
    variants and escalates temperature on retries (original behaviour).

    When *include_resolution* is False, resolution fields are stripped
    from the output (set to None) while the agent still reasons
    internally about conclusions.
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    variant_records: List[Dict[str, Any]] = []
    all_finished = True
    single_mode = num_variants == 1

    for vid in range(1, num_variants + 1):
        last_rec: Optional[Dict[str, Any]] = None
        temp_offset = 0.0

        attempts = 1 if single_mode else max_variant_attempts
        for attempt in range(attempts):
            retry_hint = ""
            if not single_mode and attempt > 0:
                retry_hint = (
                    f"\n\nRETRY NOTE (attempt {attempt + 1}/{max_variant_attempts}): "
                    "Your last attempt was too similar to earlier variants. "
                    "You MUST significantly change: (1) at least one key "
                    "question/assumption, and (2) the final resolution "
                    "type/steps. Do NOT reuse the same phrasing."
                )
                temp_offset = min(0.45, temp_offset + 0.15)

            if single_mode:
                prior_brief = "(none)"
            else:
                prior_brief = _build_prior_variants_brief(variant_records) + retry_hint

            # Build components for this attempt
            env = Environment(scenario, llm_provider)

            adjusted_temp = llm_provider.temperature + temp_offset
            agent_prov = LLMProvider(
                model=llm_provider.model,
                temperature=min(1.0, adjusted_temp),
                max_tokens=llm_provider.max_tokens,
                top_p=llm_provider.top_p,
                fallback_model=llm_provider.fallback_model,
                academic_ai_client=llm_provider.academic_ai_client,
            )

            agent = LLMAgent(
                llm_provider=agent_prov,
                scenario=scenario,
                use_native_tools=use_native_tools,
                single_mode=single_mode,
            )
            customer = LLMCustomer(
                llm_provider=agent_prov,
                scenario=scenario,
                single_mode=single_mode,
            )

            orch = Orchestrator(
                agent=agent,
                customer=customer,
                environment=env,
                scenario=scenario,
                variant_id=vid,
                agent_persona=agent_persona,
                max_turns=max_turns,
                max_errors=max_errors,
                sleep_between_turns=sleep_between_turns,
            )

            rec = orch.run(prior_variants_brief=prior_brief)
            last_rec = rec

            # Single mode: no dedup needed
            if single_mode:
                break

            # Accept unfinished variants (can't reliably compare)
            if not rec.get("finished", False):
                break

            # Deduplication check
            if not _is_too_similar(rec, variant_records):
                break

        if last_rec is not None:
            variant_records.append(last_rec)
            all_finished = all_finished and bool(last_rec.get("finished", False))

    return _flatten_to_evaluator_format(
        scenario_id=scenario_id,
        scenario=scenario,
        variant_records=variant_records,
        agent_persona=agent_persona,
        all_finished=all_finished,
        include_resolution=include_resolution,
    )


# ---------------------------------------------------------------------------
# Output formatter (evaluator-compatible JSONL columns)
# ---------------------------------------------------------------------------

def _flatten_to_evaluator_format(
    scenario_id: str,
    scenario: Dict[str, Any],
    variant_records: List[Dict[str, Any]],
    agent_persona: str,
    all_finished: bool,
    include_resolution: bool = True,
) -> Dict[str, Any]:
    """Flatten variant records into the column layout expected by evaluator.py.

    The evaluator iterates with ``while f"conversation_{{conv_idx}}" in transcript``
    so we must produce ``conversation_1``, ``conversation_2``, etc.

    When *include_resolution* is False, ``resolution_{vid}`` is set to None
    (the agent still reasons internally, but the output omits the resolution).
    """
    conv_type = "single" if len(variant_records) == 1 else "multitype"
    out: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "conversation_type": conv_type,
        "finished": all_finished,
        "source_scenario": scenario,
    }

    for rec in variant_records:
        vid = rec["conversation_variant_id"]
        out[f"conversation_{vid}"] = rec.get("conversation_history", [])
        out[f"tool_interactions_{vid}"] = rec.get("tool_interactions", [])
        out[f"customer_tool_interactions_{vid}"] = rec.get("customer_tool_interactions", [])
        out[f"customer_withdrew_{vid}"] = rec.get("customer_withdrew", False)
        out[f"agent_persona_{vid}"] = rec.get("agent_persona", agent_persona)

        agent_obj = rec.get("agent_final_object") or {}
        if isinstance(agent_obj, dict):
            out[f"resolution_{vid}"] = (
                agent_obj.get("final_resolution") if include_resolution else None
            )
            out[f"agent_summary_{vid}"] = agent_obj.get("reasoning_summary")
            out[f"agent_facts_{vid}"] = agent_obj.get("facts_collected_or_assumed")
        else:
            out[f"resolution_{vid}"] = None
            out[f"agent_summary_{vid}"] = None
            out[f"agent_facts_{vid}"] = None

    out["num_conversations"] = len(variant_records)
    return out
