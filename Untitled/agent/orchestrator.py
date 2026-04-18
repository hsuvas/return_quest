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
from typing import Any, Dict, List, Optional

from .agent import AgentInterface, LLMAgent, LLMCustomer
from .conversation_state import ConversationState, Resolution, ToolCallRecord
from .environment import Environment
from .llm_provider import LLMProvider
from .response_parser import validate_tool_call
from .tool_registry import get_agent_tools, get_customer_tools, get_tool_names


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
    return _normalize_text(
        (res.get("resolution_type", "") or "")
        + "\n"
        + (res.get("resolution_description", "") or "")
    )


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
        lines.append(
            f"- Variant {vid}: resolution={_normalize_text(res_desc)[:200]}"
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

        _MAX_TOOL_ROUNDS = 5  # max consecutive tool-call rounds before handing off

        for turn_i in range(self.max_turns + 1):
            # ---- AGENT TURN ----
            # Inner loop: let the agent call tools and then produce a message
            # in the same logical "turn" before the customer responds.
            # Without this, native-tool-calling models get stuck calling tools
            # forever because the customer turn fires before the agent replies.
            agent_error = False
            agent_produced_message = False
            for _round in range(_MAX_TOOL_ROUNDS):
                try:
                    agent_resp = self.agent.generate_response(
                        state=self.state,
                        prior_variants_brief=prior_variants_brief,
                    )
                except Exception as e:
                    print(f"  [Orchestrator] Agent error (turn {turn_i}): {e}")
                    self.state.error_count += 1
                    agent_error = True
                    break

                # Process tool calls → route through environment
                if agent_resp.tool_calls:
                    any_new = False
                    for tc in agent_resp.tool_calls:
                        if not validate_tool_call(tc, self._valid_agent_tools):
                            self.state.error_count += 1
                            continue
                        was_new = self.state.append_tool_call(tc, caller="agent")
                        if was_new:
                            any_new = True
                            tool_result = self.environment.execute_tool(tc)
                            self.state.append_tool_result(tool_result)
                    # All calls were duplicates — agent has all results in history.
                    # Mark as if a message was produced so the outer loop moves to
                    # the customer turn instead of retrying the agent indefinitely.
                    if not any_new:
                        agent_produced_message = True
                        break

                # Process agent message
                if agent_resp.message:
                    self.state.append_agent_message(agent_resp.message)
                    agent_produced_message = True

                # Update metadata from agent response
                if agent_resp.facts:
                    self.state.agent_facts = agent_resp.facts
                if agent_resp.reasoning_summary:
                    self.state.agent_summary = agent_resp.reasoning_summary
                if agent_resp.agent_persona_type:
                    self.state.agent_persona_type = agent_resp.agent_persona_type

                # Check conclusion
                if agent_resp.conclusion_reached and agent_resp.resolution:
                    pr_result = self._execute_process_return(agent_resp.resolution)
                    if isinstance(pr_result, dict) and pr_result.get("status") == "verification_required":
                        # Verification failed — re-enter inner loop so the agent
                        # reads the discrepancies and asks the customer to clarify.
                        continue
                    # Verification passed (or already attempted) — finalise.
                    self.state.resolution = agent_resp.resolution
                    if pr_result:
                        # Append a final confirmation message so the customer
                        # receives the return outcome (label URL, refund timeline).
                        confirmation = pr_result.get(
                            "message",
                            "Your return has been successfully initiated.",
                        )
                        label_url = pr_result.get("return_label_url", "")
                        if label_url:
                            confirmation += f" Return label: {label_url}"
                        self.state.append_agent_message(confirmation)
                    self.state.finished = True
                    break

                # Exit tool loop once agent sends a message or makes no tool calls
                if agent_resp.message or not agent_resp.tool_calls:
                    break
                # Otherwise: only tool calls so far → loop back with results in history

            if self.state.finished:
                break

            if agent_error:
                if self.state.error_count >= self.max_errors:
                    break
                continue  # retry agent next outer iteration, skip customer turn

            if not agent_produced_message:
                continue  # agent only made tool calls — skip customer, retry agent

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

            # Append customer message and accumulate revealed facts
            if cust_resp.reply:
                self.state.append_customer_message(cust_resp.reply)
            if cust_resp.information_provided:
                self.state.revealed_facts.extend(cust_resp.information_provided)

            # Check customer withdrawal
            if cust_resp.withdraw:
                self.state.customer_withdrew = True
                self.state.finished = True
                break

            if self.sleep_between_turns > 0:
                time.sleep(self.sleep_between_turns)

        return self._build_variant_record()

    # ----- helpers -----

    _ORDER_TOOLS = frozenset({"get_order_details", "customer_view_order_details"})

    def _extract_order_id_from_history(self) -> Optional[str]:
        """Scan history for an order_id returned by any order lookup tool."""
        for turn in self.state.history:
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    result = turn.tool_result.result
                    if "order_id" in result:
                        oid = str(result["order_id"])
                        if oid != self.scenario.get("scenario_id", ""):
                            return oid
                    orders = result.get("orders", [])
                    if orders and "order_id" in orders[0]:
                        return str(orders[0]["order_id"])
        return None

    def _extract_customer_id_from_history(self) -> Optional[str]:
        """Scan history for a system customer_id (e.g. CUST-XXXXX) from order lookups."""
        for turn in self.state.history:
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    cid = str(turn.tool_result.result.get("customer_id", ""))
                    # Accept only IDs that contain digits or hyphens (system IDs),
                    # not plain customer names.
                    if cid and re.search(r"[0-9]", cid):
                        return cid
        return None

    def _extract_items_from_history(self) -> list:
        """Extract return items from the most recent order lookup result."""
        for turn in reversed(self.state.history):
            if turn.turn == "tool_result" and turn.tool_result:
                if turn.tool_result.tool_name in self._ORDER_TOOLS:
                    result = turn.tool_result.result
                    raw_items = result.get("items") or result.get("order_items") or []
                    items = []
                    for item in raw_items:
                        if not isinstance(item, dict):
                            continue
                        item_id = (
                            item.get("item_id")
                            or item.get("product_id")
                            or item.get("sku")
                            or item.get("item_name")
                            or ""
                        )
                        items.append({
                            "item_id": str(item_id),
                            "quantity": int(item.get("quantity", 1)),
                            "condition": "unknown",
                        })
                    if items:
                        return items
        return []

    def _execute_process_return(self, resolution: Resolution) -> Optional[Dict[str, Any]]:
        """Call process_return through the environment with the resolution type.

        On the first call, runs an LLM-based verification of customer facts against
        the scenario ground truth. If verification fails, a synthetic
        ``verification_required`` result is appended to history (bypassing dedup so a
        retry can proceed) and returned to the caller to signal no-finalise.

        On subsequent calls (``state.verification_attempted=True``) verification is
        skipped and the call executes normally.

        Returns the result dict on success, the verification-failure dict when blocked,
        or None if the tool was already executed (dedup).
        """
        order_id = self._extract_order_id_from_history()
        if not order_id:
            task = self.scenario.get("task", {})
            order_id = (
                task.get("order_id")
                or task.get("order_number")
                or self.scenario.get("scenario_id", "UNKNOWN")
            )

        customer_id = (
            self._extract_customer_id_from_history()
            or self.scenario.get("persona", {}).get("customer_id")
            or self.scenario.get("persona", {}).get("Name", "UNKNOWN")
        )

        tc = ToolCallRecord(
            tool_name="process_return",
            tool_call_id=f"process_return_{self.variant_id}",
            arguments={
                "order_id": str(order_id),
                "customer_id": str(customer_id),
                "resolution_type": resolution.resolution_type,
                "items_to_return": self._extract_items_from_history(),
                "return_reason": "other",
                "return_reason_details": resolution.resolution_description,
            },
        )

        # --- VERIFICATION GATE (first attempt only) ---
        if not self.state.verification_attempted:
            self.state.verification_attempted = True
            verification = self.environment.verify_return(
                history=self.state.get_history_dicts(),
                arguments=tc.arguments,
            )
            if not verification.get("verified", True):
                # Inject tool call + synthetic result into history WITHOUT going
                # through append_tool_call so the dedup set is not touched and
                # the retry can proceed unblocked.
                from .conversation_state import ConversationTurn, ToolResultRecord
                self.state.history.append(
                    ConversationTurn(turn="tool_call", tool_calls=[tc])
                )
                self.state.tool_interactions.append({
                    "tool_call_id": tc.tool_call_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "caller": "agent",
                })
                hints = verification.get("verification_hints", [])
                synthetic_result = ToolResultRecord(
                    tool_call_id=tc.tool_call_id,
                    tool_name="process_return",
                    result={
                        "status": "verification_required",
                        "verified": False,
                        "discrepancies": verification.get("discrepancies", []),
                        "verification_hints": hints,
                        "message": (
                            "Return could not be processed: the information provided "
                            "does not match our records. Please clarify the following "
                            "with the customer before resubmitting: "
                            + ("; ".join(hints) if hints else "see discrepancies above")
                        ),
                    },
                )
                self.state.append_tool_result(synthetic_result)
                return synthetic_result.result

        # Verification passed or already attempted — execute normally
        was_new = self.state.append_tool_call(tc, caller="agent")
        if was_new:
            tool_result = self.environment.execute_tool(tc)
            self.state.append_tool_result(tool_result)
            return tool_result.result
        return None

    def _seed_customer_opening(self) -> str:
        opening = self.scenario.get("first_customer_message", "")
        if opening and isinstance(opening, str):
            return opening
        return self.scenario.get("task", {}).get("task", "")

    def _build_variant_record(self) -> Dict[str, Any]:
        # Build a consolidated resolution dict: only resolution_type and an
        # enriched resolution_description are preserved.  Fields not read by
        # the evaluator (resolution_id, conditions, customer_next_steps) and
        # separate agent-metadata fields (agent_facts, reasoning_summary) are
        # folded into resolution_description so the judge has full context in
        # the one field it actually uses.
        resolution_dict = None
        if self.state.resolution:
            base = self.state.resolution.model_dump()
            desc_parts = [base.get("resolution_description", "")]
            if base.get("conditions"):
                desc_parts.append("CONDITIONS: " + "; ".join(base["conditions"]))
            if base.get("customer_next_steps"):
                desc_parts.append("CUSTOMER NEXT STEPS: " + base["customer_next_steps"])
            if self.state.agent_facts:
                desc_parts.append("FACTS USED: " + "; ".join(self.state.agent_facts))
            if self.state.agent_summary:
                desc_parts.append("AGENT REASONING: " + self.state.agent_summary)
            resolution_dict = {
                "resolution_type": base["resolution_type"],
                "resolution_description": "\n\n".join(p for p in desc_parts if p.strip()),
            }

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
                "final_resolution": resolution_dict,
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
        out[f"resolution_{vid}"] = (
            agent_obj.get("final_resolution")
            if include_resolution and isinstance(agent_obj, dict)
            else None
        )

    out["num_conversations"] = len(variant_records)
    return out
