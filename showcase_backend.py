"""
showcase_backend.py
-------------------
Helper layer between the Streamlit UI and the existing src/agent/ pipeline.

Responsibilities:
- Load product and persona data from disk
- Build a synthetic agent-compatible scenario dict from user selections
- Run one agent turn given the current ConversationState
- Optionally generate a suggested customer message via LLMCustomer
"""

import ast
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable so `agent` package resolves
# ---------------------------------------------------------------------------

_SHOWCASE_DIR = Path(__file__).resolve().parent

if str(_SHOWCASE_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOWCASE_DIR))

from agent.agent import LLMAgent, LLMCustomer  # noqa: E402
from agent.conversation_state import ConversationState, ToolCallRecord  # noqa: E402
from agent.environment import Environment  # noqa: E402
from agent.llm_provider import LLMProvider  # noqa: E402
from agent.prompt_builder import build_customer_messages  # noqa: E402
from agent.response_parser import AgentResponse, parse_customer_response, validate_tool_call  # noqa: E402
from agent.tool_registry import get_agent_tools, get_tool_names, get_customer_tools  # noqa: E402


# ---------------------------------------------------------------------------
# Static data paths
# ---------------------------------------------------------------------------

_DATA_ROOT = _SHOWCASE_DIR / "data"
_PRODUCTS_PATH = _DATA_ROOT / "product_details" / "product_descriptions_cleaned.jsonl"
_PERSONAS_PATH = _DATA_ROOT / "persona_hub" / "example_personas.json"
_AMBIGUITIES_CSV = _SHOWCASE_DIR / "policy_ambiguities_v3_final.csv"


# ---------------------------------------------------------------------------
# Cached Amazon return policy text (condensed but accurate)
# ---------------------------------------------------------------------------

AMAZON_RETURN_POLICY_TEXT = """
Amazon Return Policy — Key Provisions

STANDARD RETURN WINDOW
Most items sold and fulfilled by Amazon can be returned within 30 days of delivery for a full refund. The 30-day return window begins on the delivery date.

THIRD-PARTY SELLER ITEMS
Items sold by third-party sellers (Marketplace sellers) may have different return policies. Amazon's A-to-z Guarantee still applies. Third-party sellers set their own return windows (minimum 30 days from delivery for most categories). If a third-party seller doesn't accept returns, customers can file an A-to-z Guarantee claim.

DEFECTIVE OR DAMAGED ITEMS
Items that arrive defective, damaged, or not as described qualify for return or replacement regardless of the standard return window, within a reasonable time frame. Customers should contact Amazon within 30 days of receiving a defective item.

REFUND METHOD
Refunds are generally issued to the original payment method. If the original payment method is unavailable, Amazon may issue a gift card. Refunds typically take 3–5 business days for credit/debit cards and up to 10 business days for bank transfers.

PARTIAL REFUNDS
A partial refund may be issued if the item is returned in a different condition than it was received, or if a non-refundable component (e.g., restocking fee for certain electronics) applies.

CATEGORIES WITH SPECIAL RULES
- Electronics & computers: 30-day return window; may have restocking fees if opened
- Hazardous materials: Non-returnable
- Digital products: Non-returnable once downloaded
- Perishables / grocery: Non-returnable
- Personalized items: Non-returnable

RETURN SHIPPING
For items sold by Amazon, return shipping is free via prepaid label. For third-party sellers, return shipping costs depend on the seller's policy. Amazon covers return shipping for defective or mis-shipped items regardless of seller.

CONDITION REQUIREMENTS
Items should be returned in original condition, in original packaging. Items showing signs of use, missing accessories, or damaged packaging may receive a partial refund or be refused.

EXCHANGE / REPLACEMENT
Amazon offers replacements for defective or damaged items at no cost. For standard returns, customers can reorder after receiving the refund.

REFUND TIMING
- Credit card: 3–5 business days after return is processed
- Amazon gift card: 2–3 hours
- Bank transfer: 5–10 business days
- Check: 10 business days

ESCALATION
If a standard return request is denied or disputed, customers can escalate to the Amazon Customer Service team or file an A-to-z Guarantee claim for third-party seller purchases.
""".strip()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_products(max_items: int = 983) -> List[Dict[str, Any]]:
    """Load product records from the cleaned JSONL file."""
    products: List[Dict[str, Any]] = []
    with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
                if len(products) >= max_items:
                    break
    return products


def load_products_balanced(n: int = 50) -> List[Dict[str, Any]]:
    """Return n products sampled with balanced top-level category distribution."""
    import math
    from collections import defaultdict

    all_products = load_products()
    buckets: Dict[str, List] = defaultdict(list)
    for p in all_products:
        top = (p.get("category") or "").split("|")[0].strip() or "Other"
        buckets[top].append(p)

    cats = list(buckets.keys())
    per_cat = math.ceil(n / len(cats))
    rng = random.Random(42)  # fixed seed — consistent across sessions

    result: List[Dict[str, Any]] = []
    for cat in cats:
        picks = buckets[cat]
        if len(picks) > per_cat:
            picks = rng.sample(picks, per_cat)
        result.extend(picks)

    # shuffle so categories interleave, then trim
    rng.shuffle(result)
    return result[:n]


def load_personas() -> List[Dict[str, Any]]:
    """Load preset personas from the JSON file."""
    with open(_PERSONAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_product_categories(products: List[Dict[str, Any]]) -> List[str]:
    """Return sorted unique top-level categories."""
    cats = set()
    for p in products:
        cat = p.get("category", "")
        if cat:
            top = cat.split("|")[0].strip()
            if top:
                cats.add(top)
    return sorted(cats)


# ---------------------------------------------------------------------------
# Persona sector → emoji mapping
# ---------------------------------------------------------------------------

SECTOR_EMOJI: Dict[str, str] = {
    "Technology": "💻",
    "Education": "📚",
    "Academia": "🎓",
    "Healthcare": "🏥",
    "Finance": "💰",
    "Law": "⚖️",
    "Art": "🎨",
    "Engineering": "🔧",
    "Science": "🔬",
    "Business": "💼",
    "Retail": "🛍️",
    "Government": "🏛️",
    "Media": "📺",
    "Sports": "⚽",
}


def persona_emoji(persona: Dict[str, Any]) -> str:
    sector = persona.get("Job Sector", "")
    for key, emoji in SECTOR_EMOJI.items():
        if key.lower() in sector.lower():
            return emoji
    return "👤"


# ---------------------------------------------------------------------------
# Resolution type → display mapping
# ---------------------------------------------------------------------------

RESOLUTION_DISPLAY: Dict[str, Dict[str, str]] = {
    "RETURN_REFUND_FULL_BANK": {
        "label": "Full Refund to Bank Account",
        "icon": "💰",
        "color": "green",
        "description": "You will receive a full refund to your original payment method within 3–5 business days.",
    },
    "RETURN_REFUND_PARTIAL_BANK": {
        "label": "Partial Refund to Bank Account",
        "icon": "💳",
        "color": "orange",
        "description": "A partial refund has been approved. Deductions may apply for condition or restocking fees.",
    },
    "RETURN_REFUND_GIFT_CARD": {
        "label": "Store Credit / Gift Card",
        "icon": "🎁",
        "color": "blue",
        "description": "You will receive an Amazon Gift Card for the refund amount. This is typically processed within 2–3 hours.",
    },
    "DENY_REFUND": {
        "label": "Return Denied",
        "icon": "❌",
        "color": "red",
        "description": "Your return request has been denied based on the applicable policy. See details below.",
    },
    "ESCALATE_HUMAN_AGENT": {
        "label": "Escalated to Human Specialist",
        "icon": "👤",
        "color": "purple",
        "description": "Your case has been escalated to a specialist. You will be contacted within 1–2 business days.",
    },
    "REPLACEMENT_EXCHANGE": {
        "label": "Replacement / Exchange",
        "icon": "🔄",
        "color": "teal",
        "description": "A replacement or exchange has been arranged. See next steps for details.",
    },
    "USER_ABORT": {
        "label": "Request Withdrawn",
        "icon": "🚪",
        "color": "gray",
        "description": "The return request was withdrawn. You can start a new request at any time.",
    },
}


# ---------------------------------------------------------------------------
# Policy issue derivation
# ---------------------------------------------------------------------------

_REASON_ISSUE_MAP: Dict[str, str] = {
    "Defective": (
        "Defective item return: standard 30-day window may conflict with "
        "manufacturer defect discovery timeline"
    ),
    "Wrong Item": (
        "Wrong item fulfillment: responsibility ambiguity between Amazon "
        "and third-party seller for mispicks"
    ),
    "Changed Mind": (
        "Change-of-mind return: item must be in original, unopened condition "
        "— condition assessment is subjective"
    ),
    "Damaged in Shipping": (
        "Carrier-damaged item: return vs. shipping insurance claim path "
        "is ambiguous for third-party seller items"
    ),
    "Other": (
        "Non-standard return reason: agent discretion required; "
        "no clear policy guidance"
    ),
}

_THIRD_PARTY_ISSUE = (
    "Third-party seller policy: seller's return window and conditions may "
    "differ from Amazon's standard policy"
)


def derive_policy_issues(
    items: List[Dict[str, Any]],
    reasons: List[str],
) -> List[str]:
    """Derive a list of policy tension strings from items and return reasons."""
    issues = []
    has_third_party = any(
        item.get("is_amazon_seller", "Y") != "Y" for item in items
    )
    if has_third_party:
        issues.append(_THIRD_PARTY_ISSUE)

    seen_reasons = set()
    for reason in reasons:
        if reason not in seen_reasons:
            seen_reasons.add(reason)
            issue = _REASON_ISSUE_MAP.get(reason, _REASON_ISSUE_MAP["Other"])
            if issue not in issues:
                issues.append(issue)

    return issues


# ---------------------------------------------------------------------------
# LLM task generation
# ---------------------------------------------------------------------------

_TASK_SYS_PROMPT = (
    "You are an expert at designing high-complexity Amazon order return scenarios "
    "for customer support training. Follow the instructions exactly and return only JSON."
)

_TASK_PROMPT_TEMPLATE = """\
## INSTRUCTION
Generate exactly 1 high-complexity order return task that:
- Uses the provided product set as the purchased items
- Exploits the provided policy ambiguities (do NOT resolve them)
- Requires multi-turn clarification between customer and agent
- Has at least 2 plausible resolution outcomes

## POLICY
{policy_text}

## POLICY AMBIGUITIES (exploit these)
{ambiguities_json}

## PRODUCT SET
{products_json}

## CUSTOMER CONTEXT
Customer name: {customer_name}
Order date: {order_date}
Delivery date: {delivery_date}

## OUTPUT FORMAT
Return a single JSON object (not an array) with these fields:
- "detail": Long paragraph describing the order context, timeline, item conditions, and ambiguous situation. Do NOT reference "task_1_products" or internal names.
- "related_policy_issues": List of 3-5 specific policy tensions this task exploits (short phrases).
- "complexity_level": One of "Medium Complexity" | "High Complexity" | "Very High Complexity"

Output ONLY the JSON object, no markdown fences.
"""


def _load_ambiguities(n: int = 4) -> List[Dict]:
    """Load n random ambiguities from the CSV."""
    df = pd.read_csv(_AMBIGUITIES_CSV)
    sample = df.sample(min(n, len(df)))
    result = []
    for _, row in sample.iterrows():
        raw = row.get("List_of_ambiguities", "[]")
        try:
            lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            lst = [str(raw)]
        result.append({
            "policy_clause": row.get("policy_clause", ""),
            "Ambiguity_type": row.get("Ambiguity_type", ""),
            "Ambiguity_description": row.get("Ambiguity_description", ""),
            "List_of_ambiguities": lst,
        })
    return result


def generate_task_detail(
    selected_items: List[Dict[str, Any]],
    persona: Dict[str, Any],
    provider: LLMProvider,
) -> Dict[str, Any]:
    """Call the LLM to create a complex task from selected products + random ambiguities.

    Returns a dict with keys: detail, related_policy_issues, complexity_level.
    Falls back gracefully on any error.
    """
    import datetime
    ambiguities = _load_ambiguities(n=4)

    # Synthesise plausible order/delivery dates
    today = datetime.date.today()
    order_date = (today - datetime.timedelta(days=random.randint(20, 35))).strftime("%B %d, %Y")
    delivery_date = (today - datetime.timedelta(days=random.randint(8, 18))).strftime("%B %d, %Y")

    products_simple = [
        {"product_name": p.get("product_name", ""), "description": p.get("description", "")}
        for p in selected_items
    ]

    prompt = _TASK_PROMPT_TEMPLATE.format(
        policy_text=AMAZON_RETURN_POLICY_TEXT[:3000],  # trim to keep tokens manageable
        ambiguities_json=json.dumps(ambiguities, ensure_ascii=False, indent=2),
        products_json=json.dumps(products_simple, ensure_ascii=False, indent=2),
        customer_name=persona.get("Name", "the customer"),
        order_date=order_date,
        delivery_date=delivery_date,
    )

    messages = [
        {"role": "system", "content": _TASK_SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        resp = provider.call_text_only(messages=messages, temperature=0.9, max_tokens=900)
        text = (resp.content or "{}").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        parsed = json.loads(text)
        parsed["order_date"] = order_date
        parsed["delivery_date"] = delivery_date
        parsed["ambiguities_used"] = ambiguities
        return parsed
    except Exception as e:
        return {
            "detail": "",
            "related_policy_issues": [],
            "complexity_level": "High Complexity",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(
    persona: Dict[str, Any],
    selected_items: List[Dict[str, Any]],
    return_reasons: List[str],
    first_message: Optional[str] = None,
    task_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a synthetic agent-compatible scenario dict.

    If task_detail (from generate_task_detail) is provided, its richer
    description and policy issues are merged into the task field.
    """
    order_id = f"AMZ-{random.randint(1000000, 9999999)}"

    items_text = ", ".join(
        f"{item['product_name']} ({item.get('selling_price', 'N/A')})"
        for item in selected_items
    )
    reason_text = "; ".join(
        f"{item['product_name']}: {reason}"
        for item, reason in zip(selected_items, return_reasons)
    )

    policy_issues = (
        task_detail.get("related_policy_issues") or derive_policy_issues(selected_items, return_reasons)
        if task_detail else derive_policy_issues(selected_items, return_reasons)
    )

    task_description = (task_detail or {}).get("detail") or (
        f"Customer {persona.get('Name', 'Customer')} wants to return: {items_text}. "
        f"Return reasons: {reason_text}. Order ID: {order_id}."
    )

    purchase_date = (task_detail or {}).get("order_date", "")
    delivery_date = (task_detail or {}).get("delivery_date", "")

    # CRM-visible product names (what the agent simulator can see via get_order_details)
    products_involved = [item["product_name"] for item in selected_items]

    # Seller type for the case brief
    has_third_party = any(
        str(item.get("is_amazon_seller", "Y")).strip().upper() != "Y"
        for item in selected_items
    )
    seller_note = "Third-party seller (Fulfilled by Amazon)" if has_third_party else "Sold and fulfilled by Amazon"

    # Short agent-visible case brief: order facts only, NO conditions or return reasons.
    # This is what the new prompt_builder injects as {detail_agent}.
    detail_agent = (
        f"Order ID: {order_id}\n"
        f"Items: {', '.join(products_involved)}\n"
        + (f"Order date: {purchase_date}\n" if purchase_date else "")
        + (f"Delivery date: {delivery_date}\n" if delivery_date else "")
        + f"Seller: {seller_note}\n"
        f"Customer: {persona.get('Name', 'Customer')}"
    )

    return {
        "scenario_id": f"demo_{uuid.uuid4().hex[:8]}",
        "Policy": {
            "Primary Policy": {
                "url": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7",
                "text": AMAZON_RETURN_POLICY_TEXT,
            },
            "Related policies": [],
        },
        "persona": persona,
        # Short agent-visible brief (no conditions or return reasons)
        "detail_agent": detail_agent,
        "task": {
            "order_id": order_id,
            "order_date": purchase_date,          # matches _ORDER_FACT_KEYS
            "delivery_date": delivery_date,
            "products_involved": products_involved,  # matches _ORDER_FACT_KEYS
            "items": selected_items,              # full product dicts for environment
            "return_reasons": dict(zip(
                [item["product_name"] for item in selected_items],
                return_reasons,
            )),
            "task": task_description,
            "detail": task_description,
            "related_policy_issues": policy_issues,
            "complexity_level": (task_detail or {}).get("complexity_level", "High Complexity"),
            "purchase_date": purchase_date,
        },
        "first_customer_message": first_message,
    }


# ---------------------------------------------------------------------------
# LLM Provider factory
# ---------------------------------------------------------------------------

def make_provider(
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2500,
) -> LLMProvider:
    """Create an LLMProvider using OPENAI_API_KEY from the environment."""
    return LLMProvider(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Conversation state factory
# ---------------------------------------------------------------------------

def make_conversation_state(
    scenario: Dict[str, Any],
    agent_persona: str,
) -> ConversationState:
    return ConversationState(
        scenario=scenario,
        variant_id=1,
        agent_persona=agent_persona,
    )


# ---------------------------------------------------------------------------
# Single agent turn
# ---------------------------------------------------------------------------

def run_agent_turn(
    state: ConversationState,
    scenario: Dict[str, Any],
    provider: LLMProvider,
    agent_persona: str,
) -> Tuple[AgentResponse, List[Dict[str, Any]]]:
    """
    Run one agent turn.

    Returns:
        (agent_response, tool_results_list)

    The caller is responsible for appending the customer message to state
    *before* calling this function.
    """
    # use_native_tools=False: agent outputs a single JSON blob containing both
    # tool_calls_made and conversation_flow[0].message in one LLM call.
    # This avoids the native-tools bug where the LLM sends tool calls with
    # empty content, leaving agent_resp.message = None.
    agent = LLMAgent(
        llm_provider=provider,
        scenario=scenario,
        use_native_tools=False,
        single_mode=True,
    )
    env = Environment(scenario=scenario, llm_provider=provider)
    valid_tools = get_tool_names(get_agent_tools())

    tool_results: List[Dict[str, Any]] = []
    agent_resp = None

    # Multi-round inner loop: execute tools, then call agent again so its
    # customer-visible message is written AFTER seeing tool results.
    _MAX_TOOL_ROUNDS = 5
    for _round in range(_MAX_TOOL_ROUNDS):
        agent_resp = agent.generate_response(state=state)

        executed_new = False
        if agent_resp.tool_calls:
            for tc in agent_resp.tool_calls:
                if not validate_tool_call(tc, valid_tools):
                    continue
                was_new = state.append_tool_call(tc, caller="agent")
                if was_new:
                    tr = env.execute_tool(tc)
                    state.append_tool_result(tr)
                    tool_results.append({
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "result": tr.result,
                    })
                    executed_new = True

        if executed_new:
            # Loop back: agent will now see tool results and write a real reply
            continue
        break  # No new tools executed — agent_resp.message is the final reply

    # Append agent message — always use fallback if LLM produced nothing
    message = agent_resp.message or "I'm looking into your request. Could you give me a moment?"
    state.append_agent_message(message)

    # Update state metadata from the final agent response
    if agent_resp.facts:
        state.agent_facts = agent_resp.facts
    if agent_resp.reasoning_summary:
        state.agent_summary = agent_resp.reasoning_summary
    if agent_resp.agent_persona_type:
        state.agent_persona_type = agent_resp.agent_persona_type

    # Mark finished if conclusion reached
    if agent_resp.conclusion_reached and agent_resp.resolution:
        state.resolution = agent_resp.resolution
        state.finished = True

    return message, tool_results, agent_resp


# ---------------------------------------------------------------------------
# Suggest first customer message
# ---------------------------------------------------------------------------

def suggest_first_message(
    scenario: Dict[str, Any],
    provider: LLMProvider,
) -> str:
    """
    Use LLMCustomer to suggest an opening message for the customer.
    We seed the history with a synthetic "Hello" from the agent so the
    customer prompt has a latest_agent_message to respond to.
    """
    state = ConversationState(
        scenario=scenario,
        variant_id=1,
        agent_persona="FAIR",
    )
    # Inject a synthetic agent opener so the customer prompt works
    state.append_agent_message(
        "Hello! Welcome to Amazon Customer Service. How can I assist you today? "
        "Please describe your return request including the item(s) you'd like to return "
        "and the reason for the return."
    )

    customer = LLMCustomer(
        llm_provider=provider,
        scenario=scenario,
        single_mode=True,
    )
    try:
        resp = customer.generate_response(state=state)
        return resp.reply
    except Exception as e:
        # Fallback template
        persona = scenario.get("persona", {})
        task = scenario.get("task", {})
        order_id = task.get("order_id", "your order")
        items = task.get("items", [])
        item_name = items[0]["product_name"] if items else "the item"
        name = persona.get("Name", "Customer")
        return (
            f"Hi, I'm {name} and I need help returning {item_name}. "
            f"My order number is {order_id}. I'm hoping we can resolve this quickly."
        )


# ---------------------------------------------------------------------------
# Suggest next customer message (hint)
# ---------------------------------------------------------------------------

def suggest_next_message(
    state: ConversationState,
    scenario: Dict[str, Any],
    provider: LLMProvider,
) -> str:
    """Return a one-sentence hint telling the player exactly what to say next."""
    # Find the last agent message
    last_agent_message = ""
    for turn in reversed(state.history):
        if turn.turn == "agent" and turn.message:
            last_agent_message = turn.message
            break

    if not last_agent_message:
        return "Describe your return request and provide your order details to the agent."

    # Build task context from scenario
    task = scenario.get("task", {})
    items = task.get("items", [])
    product_names = ", ".join(i.get("product_name", "") for i in items) or "N/A"
    return_reasons = task.get("return_reasons", {})
    reasons_text = "; ".join(f"{k}: {v}" for k, v in return_reasons.items()) or "N/A"
    task_description = task.get("task", task.get("detail", ""))

    task_context = (
        f"Order ID: {task.get('order_id', 'N/A')}\n"
        f"Purchase date: {task.get('purchase_date', task.get('order_date', 'N/A'))}\n"
        f"Delivery date: {task.get('delivery_date', 'N/A')}\n"
        f"Product(s): {product_names}\n"
        f"Return reason(s): {reasons_text}\n"
        f"Task: {task_description}"
    )

    prompt = (
        "You are a coach helping a player in a customer-service role-play game. "
        "The player is the customer. "
        "Based on the agent's last message and the player's scenario facts, "
        "write ONE short sentence (max 25 words) telling the player exactly what to say or provide next. "
        "Mention the specific value, date, or detail from the scenario they should use. "
        "Be concrete — do not be vague or generic.\n\n"
        f"Agent's last message:\n{last_agent_message}\n\n"
        f"Player's scenario:\n{task_context}\n\n"
        "Hint:"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = provider.call_text_only(messages=messages, temperature=0.3, max_tokens=60)
        hint = (resp.content or "").strip().lstrip("Hint:").strip()
        return hint or "Answer the agent's question using your scenario details."
    except Exception:
        return "Answer the agent's question using your scenario details."


# ---------------------------------------------------------------------------
# Narrative + conversation starter generation (stubs; LLM call added later)
# ---------------------------------------------------------------------------

def _short_product_name(name: str, max_words: int = 3) -> str:
    """Return the first max_words words of a product name."""
    words = name.split()
    if len(words) <= max_words:
        return name
    return " ".join(words[:max_words]) + "…"


def generate_narrative(
    scenario: Dict[str, Any],
    provider: Optional[LLMProvider] = None,
    kid_mode: bool = False,
) -> str:
    """Build the mission narrative shown on Step 4.

    If the scenario has an LLM-generated task detail, display that.
    Otherwise fall back to a template.
    """
    persona = scenario.get("persona", {})
    task = scenario.get("task", {})
    name = persona.get("Name", "Customer")
    age = persona.get("Age-range", "")
    location = persona.get("Location", "")
    age_loc = f", {age} from {location}" if age and location else ""

    mission_cta = (
        '<span style="font-weight:700;font-size:1.05rem;display:block;margin-top:10px;'
        'background:#e0f2f1;border-left:4px solid #00695c;padding:6px 10px;border-radius:3px;">'
        '🎯 Your mission: negotiate the best possible outcome with the Customer Service Agent!'
        '</span>'
    )

    detail = task.get("detail", "")
    complexity = task.get("complexity_level", "")
    policy_issues = task.get("related_policy_issues", [])

    if detail:
        issues_html = ""
        if policy_issues:
            bullets = "".join(f"<li>{issue}</li>" for issue in policy_issues)
            issues_html = (
                f'<div style="margin-top:10px;font-size:0.82rem;color:#546e7a;">'
                f'<strong>Policy tensions in play:</strong><ul style="margin:4px 0 0 16px;">{bullets}</ul></div>'
            )
        complexity_html = ""
        if complexity:
            complexity_html = (
                f'<div style="margin-top:6px;font-size:0.78rem;color:#bf360c;font-weight:600;">'
                f'Difficulty: {complexity}</div>'
            )
        return (
            f"<strong>You are {name}{age_loc}.</strong><br><br>"
            f"{detail}"
            + complexity_html
            + issues_html
            + mission_cta
        )

    # Fallback template
    items = task.get("items", [])
    reasons = task.get("return_reasons", {})
    raw_item_name = items[0]["product_name"] if items else "the item"
    item_name = _short_product_name(raw_item_name, 3) if kid_mode else raw_item_name
    reason = list(reasons.values())[0] if reasons else "an issue"
    return (
        f"You are {name}{age_loc}. "
        f"You recently purchased {item_name} from Amazon, "
        f"but you need to return it because: {reason.lower()}. "
        + mission_cta
    )


def generate_starters(
    scenario: Dict[str, Any],
    provider: Optional[LLMProvider] = None,
    kid_mode: bool = False,
) -> List[str]:
    """Return 3 persona-voiced opening messages.

    Stub implementation returns templates.
    The LLM-powered version will be wired in during the backend phase.
    """
    persona = scenario.get("persona", {})
    task = scenario.get("task", {})
    order_id = task.get("order_id", "your order")
    items = task.get("items", [])
    reasons = task.get("return_reasons", {})
    raw_item_name = items[0]["product_name"] if items else "the item"
    item_name = _short_product_name(raw_item_name, 3) if kid_mode else raw_item_name
    reason = list(reasons.values())[0] if reasons else "an issue"
    name = persona.get("Name", "Customer")
    return [
        f"Hi, I need help returning my order {order_id}.",
        (
            f"Hello, I'm {name}. My order {order_id} had a problem — "
            f"the {item_name} was {reason.lower()}. Can you help?"
        ),
        (
            f"Good day. I would like to initiate a return for {item_name} "
            f"(order {order_id}). The item was {reason.lower()}."
        ),
    ]
