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
# Cached Amzaon return policy text (condensed but accurate)
# ---------------------------------------------------------------------------

AMAZON_RETURN_POLICY_TEXT = """
Amzaon Return Policy — Key Provisions

STANDARD RETURN WINDOW
Most items sold and fulfilled by Amzaon can be returned within 30 days of delivery for a full refund. The 30-day return window begins on the delivery date.

THIRD-PARTY SELLER ITEMS
Items sold by third-party sellers (Marketplace sellers) may have different return policies. Amzaon's A-to-z Guarantee still applies. Third-party sellers set their own return windows (minimum 30 days from delivery for most categories). If a third-party seller doesn't accept returns, customers can file an A-to-z Guarantee claim.

DEFECTIVE OR DAMAGED ITEMS
Items that arrive defective, damaged, or not as described qualify for return or replacement regardless of the standard return window, within a reasonable time frame. Customers should contact Amzaon within 30 days of receiving a defective item.

REFUND METHOD
Refunds are generally issued to the original payment method. If the original payment method is unavailable, Amzaon may issue a gift card. Refunds typically take 3–5 business days for credit/debit cards and up to 10 business days for bank transfers.

PARTIAL REFUNDS
A partial refund may be issued if the item is returned in a different condition than it was received, or if a non-refundable component (e.g., restocking fee for certain electronics) applies.

CATEGORIES WITH SPECIAL RULES
- Electronics & computers: 30-day return window; may have restocking fees if opened
- Hazardous materials: Non-returnable
- Digital products: Non-returnable once downloaded
- Perishables / grocery: Non-returnable
- Personalized items: Non-returnable

RETURN SHIPPING
For items sold by Amzaon, return shipping is free via prepaid label. For third-party sellers, return shipping costs depend on the seller's policy. Amzaon covers return shipping for defective or mis-shipped items regardless of seller.

CONDITION REQUIREMENTS
Items should be returned in original condition, in original packaging. Items showing signs of use, missing accessories, or damaged packaging may receive a partial refund or be refused.

EXCHANGE / REPLACEMENT
Amzaon offers replacements for defective or damaged items at no cost. For standard returns, customers can reorder after receiving the refund.

REFUND TIMING
- Credit card: 3–5 business days after return is processed
- Amzaon gift card: 2–3 hours
- Bank transfer: 5–10 business days
- Check: 10 business days

ESCALATION
If a standard return request is denied or disputed, customers can escalate to the Amzaon Customer Service team or file an A-to-z Guarantee claim for third-party seller purchases.
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
        "description": "You will receive an Amzaon Gift Card for the refund amount. This is typically processed within 2–3 hours.",
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
        "Wrong item fulfillment: responsibility ambiguity between Amzaon "
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
    "differ from Amzaon's standard policy"
)


def derive_policy_issues(
    items: List[Dict[str, Any]],
    reasons: List[str],
) -> List[str]:
    """Derive a list of policy tension strings from items and return reasons."""
    issues = []
    has_third_party = any(
        item.get("is_amzaon_seller", "Y") != "Y" for item in items
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
    "You are an expert at designing high-complexity Amzaon order return scenarios "
    "for customer support training. Follow the instructions exactly and return only JSON."
)

_TASK_PROMPT_TEMPLATE = """\
## INSTRUCTION

Generate exactly 1 high-complexity Amzaon order return scenario for a customer support negotiation game.
- Use the provided product set as the purchased items
- Exploit the provided policy ambiguities — do NOT resolve them
- Require multi-turn clarification between customer and agent
- Have at least 2 and at most 6 plausible resolution outcomes

**CRITICAL: The return reason(s) listed below MUST genuinely occur in the scenario. Do not contradict or negate them.**

## RETURN REASONS (must be factually present)
{return_reasons_text}

---

## GROUNDING IN ACTUAL RETURN PROCESSES

Scenarios MUST reflect how Amzaon's return process actually works. Key distinctions:

1. Missing items vs returns: A missing item is a delivery issue (reshipment or partial refund), not a return. Create ambiguity around edge cases such as a customer who wants to return an incomplete order rather than wait for reshipment.

2. Partial delivery: If not all items were delivered, create genuine uncertainty about whether the customer can return what they did receive, whether the return window starts from original delivery or order completion, and how a partial refund interacts with return eligibility.

3. Replacement vs exchange: Replacement means getting the SAME item again (for defective or damaged goods). Exchange means getting a DIFFERENT item (different color, size, model). Make this distinction clear and create scenarios that test it.

4. Original or unused condition edge cases: Create scenarios where the condition is genuinely ambiguous — for example, packaging opened to verify contents but product untouched, electronics with intact seals but functionality tested briefly, or clothing tried on but not worn outside.

5. Bundle vs separate items: A bundle is a product sold as a set. Items purchased together in the same order are separate products. These have different return implications and create different ambiguities.

6. Heavy or bulky items: When relevant, consider that large items may be excluded from free return shipping or standard drop-off options.

7. Third-party seller timing: When items are from third-party sellers, create tension between the seller's verification timeline and customer urgency (upcoming event, financial pressure, expiring window).

---

## TASK DESIGN REQUIREMENTS

The scenario must:
- Require several clarifying questions from the agent before a resolution is possible
- Have at least one item-level difference in condition or eligibility if multiple products are involved
- Include timeline details that create genuine uncertainty (ordered date, delivered date, when issue was discovered, how much of the return window remains)
- Leave at least one key fact ambiguous — something the agent must ask about to resolve
- Be internally consistent and realistic — no contrived or nonsensical situations
- Exploit at least 3 of the provided policy ambiguities in a way that meaningfully affects possible outcomes

**CRITICAL: The return_details narrative MUST reflect the customer's actual return reasons as stated below.
Each return reason must genuinely occur in the scenario — do not contradict or negate these reasons.**

---

## POLICY
{policy_text}

## POLICY AMBIGUITIES (exploit these — do NOT mention policy by name in any narrative field)
{ambiguities_json}

## PRODUCT SET
{products_json}

## CUSTOMER CONTEXT
Customer name: {customer_name}
Order date: {order_date}
Delivery date: {delivery_date}
Pre-assigned order IDs: {order_ids_text}
If all items were shipped together, use only the first order ID. If items were ordered or shipped separately, assign each a distinct order ID from the list above.

---

## OUTPUT FORMAT

Return a single JSON object (not an array) with exactly these fields:

- "detail": A long internal case description (6–10 sentences) written from the perspective of an internal case file.
  Include: items purchased and their quantities, full order timeline (ordered, delivered, issue discovered, current date relative to return window), delivery status, item-level conditions with specifics (sealed, opened, used, damaged), any prior communication with the seller, customer circumstances that create urgency or complicate resolution, and the specific facts that leave the correct outcome genuinely unclear.
  Write as an internal briefing — not a customer message and not addressed to the customer.
  Do NOT reference product set names. Do NOT resolve the ambiguities. Include enough detail that an agent must ask questions.

- "return_details": A 2nd-person narrative (4–6 sentences) addressed to the customer.
  Write as: "You ordered...", "When the package arrived...", "You noticed...".
  Describe what was ordered, when it arrived, the condition of each item, what happened, and the return reason with full context including any timeline pressure.
  The return reason(s) listed above MUST be evident as real occurrences.
  Do NOT include order IDs (added separately). Do NOT reference policy rules or policy names.

- "customer_behavior": A JSON object with exactly these sub-fields:
  - "things_to_hide": List of strings — facts the customer will NOT volunteer unless the agent asks directly.
  - "things_to_reveal_if_asked": List of strings — facts disclosed only when the agent probes with a specific question.
  - "negotiation_style": One of "assertive" | "cooperative" | "evasive" | "emotional"
  - "expected_outcome": String — what the customer ideally wants.

- "related_policy_issues": List of 3–5 short phrases naming the specific policy tensions this scenario exploits. Each phrase must be directly relevant to the scenario details, not generic.

- "customer_agent_info": A 1–2 sentence summary of only what the customer openly presents as their issue. No hidden details, no policy tensions, nothing from things_to_hide. This is the surface-level statement the customer would make upfront. Example: "Customer wants to return a security camera that is incompatible with their Wi-Fi and is asking about return timing for the separately delivered extension cord."

- "complexity_level": One of "Medium Complexity" | "High Complexity" | "Very High Complexity"
  Medium: 1–2 ambiguities, a few clarifying questions.
  High: 2–3 ambiguities, several clarifying questions, possibly multiple items with different statuses.
  Very High: 3+ ambiguities, conflicting details, multiple stakeholders, time pressure, extensive clarification required.

Output ONLY the JSON object, no markdown fences, no commentary.
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
    return_reasons: Optional[List[str]] = None,
    order_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Call the LLM to create a complex task from selected products + random ambiguities.

    Returns a dict with keys: detail, related_policy_issues, complexity_level.
    Falls back gracefully on any error.
    """
    import datetime
    ambiguities = _load_ambiguities(n=5)

    # Synthesise plausible order/delivery dates
    today = datetime.date.today()
    order_date = (today - datetime.timedelta(days=random.randint(20, 35))).strftime("%B %d, %Y")
    delivery_date = (today - datetime.timedelta(days=random.randint(8, 18))).strftime("%B %d, %Y")

    products_simple = [
        {"product_name": p.get("product_name", ""), "description": p.get("description", "")}
        for p in selected_items
    ]

    reasons = return_reasons or []
    return_reasons_text = "; ".join(
        f"{item.get('product_name', 'item')}: {reason}"
        for item, reason in zip(selected_items, reasons)
    ) or "Not specified"

    if not order_ids:
        order_ids = [f"AMZ-{random.randint(1000000, 9999999)}" for _ in selected_items]
    order_ids_text = ", ".join(order_ids)

    prompt = _TASK_PROMPT_TEMPLATE.format(
        policy_text=AMAZON_RETURN_POLICY_TEXT[:3000],  # trim to keep tokens manageable
        ambiguities_json=json.dumps(ambiguities, ensure_ascii=False, indent=2),
        products_json=json.dumps(products_simple, ensure_ascii=False, indent=2),
        customer_name=persona.get("Name", "the customer"),
        order_date=order_date,
        delivery_date=delivery_date,
        return_reasons_text=return_reasons_text,
        order_ids_text=order_ids_text,
    )

    messages = [
        {"role": "system", "content": _TASK_SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]

    _DEFAULT_CUSTOMER_BEHAVIOR = {
        "things_to_hide": [],
        "things_to_reveal_if_asked": [],
        "negotiation_style": "cooperative",
        "expected_outcome": "full refund or exchange",
    }

    try:
        resp = provider.call_text_only(messages=messages, temperature=0.9, max_tokens=2500)
        text = (resp.content or "{}").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        parsed = json.loads(text)
        # LLM may still emit "detail" from old pattern — alias it
        if "return_details" not in parsed:
            parsed["return_details"] = parsed.pop("detail", "")
        # customer_behavior may arrive as a JSON string
        cb = parsed.get("customer_behavior")
        if isinstance(cb, str):
            try:
                parsed["customer_behavior"] = json.loads(cb)
            except Exception:
                parsed["customer_behavior"] = _DEFAULT_CUSTOMER_BEHAVIOR
        elif not isinstance(cb, dict):
            parsed["customer_behavior"] = _DEFAULT_CUSTOMER_BEHAVIOR
        parsed.setdefault("related_policy_issues", [])
        parsed.setdefault("complexity_level", "High Complexity")
        parsed.setdefault("customer_agent_info", "")
        parsed.setdefault("detail", "")
        parsed["order_date"] = order_date
        parsed["delivery_date"] = delivery_date
        parsed["ambiguities_used"] = ambiguities
        parsed["order_ids"] = order_ids
        return parsed
    except Exception as e:
        return {
            "return_details": "",
            "customer_behavior": _DEFAULT_CUSTOMER_BEHAVIOR,
            "related_policy_issues": [],
            "complexity_level": "High Complexity",
            "customer_agent_info": "",
            "order_ids": order_ids,
            "order_date": order_date,
            "delivery_date": delivery_date,
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
    order_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a synthetic agent-compatible scenario dict.

    If task_detail (from generate_task_detail) is provided, its richer
    description and policy issues are merged into the task field.
    """
    # Use pre-generated IDs (from generate_task_detail) or create new ones
    _ids = order_ids or (task_detail or {}).get("order_ids") or [
        f"AMZ-{random.randint(1000000, 9999999)}" for _ in selected_items
    ]
    order_id = _ids[0]  # primary (backward compat)

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

    purchase_date = (task_detail or {}).get("order_date", "")
    delivery_date = (task_detail or {}).get("delivery_date", "")

    # CRM-visible product names (what the agent simulator can see via get_order_details)
    products_involved = [item["product_name"] for item in selected_items]

    # Seller type for the case brief
    has_third_party = any(
        str(item.get("is_amzaon_seller", "Y")).strip().upper() != "Y"
        for item in selected_items
    )
    seller_note = "Third-party seller (Fulfilled by Amzaon)" if has_third_party else "Sold and fulfilled by Amzaon"

    # --- Three-part scenario structure ---

    # 1. Basic info: shared facts accessible to both sides
    basic_info = {
        "order_id": order_id,          # primary ID (backward compat)
        "order_ids": _ids,             # all order IDs (one per item for separate shipments)
        "order_date": purchase_date,
        "delivery_date": delivery_date,
        "products": [
            {"product_name": item["product_name"], "price": item.get("selling_price", "N/A")}
            for item in selected_items
        ],
        "seller": seller_note,
    }

    # 2. Return details: customer's story (agent learns gradually through conversation)
    return_details = (task_detail or {}).get("return_details") or (
        f"You purchased {items_text}. Your return reason(s): {reason_text}."
    )

    # 3. Customer behavior: simulator instructions only, never shown in UI
    _default_behavior = {
        "things_to_hide": [],
        "things_to_reveal_if_asked": [],
        "negotiation_style": "cooperative",
        "expected_outcome": "full refund or exchange",
    }
    customer_behavior = (task_detail or {}).get("customer_behavior") or _default_behavior

    order_ids_str = ", ".join(_ids) if len(_ids) > 1 else order_id

    # Combined ground truth for verify_return: prefer LLM-generated detail, fall back to programmatic
    _llm_detail = (task_detail or {}).get("detail", "").strip()
    if _llm_detail:
        combined_detail = (
            f"Order ID(s): {order_ids_str}\n"
            f"Order date: {purchase_date}\n"
            f"Delivery date: {delivery_date}\n"
            f"Seller: {seller_note}\n\n"
            f"{_llm_detail}"
        )
    else:
        combined_detail = (
            f"Order ID(s): {order_ids_str}\n"
            f"Order date: {purchase_date}\n"
            f"Delivery date: {delivery_date}\n"
            f"Items: {items_text}\n"
            f"Seller: {seller_note}\n\n"
            f"{return_details}"
        )

    # Short agent-visible case brief: order facts only, NO conditions or return reasons.
    # This is what the new prompt_builder injects as {detail_agent}.
    detail_agent = (
        f"Order ID(s): {order_ids_str}\n"
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
                "url": "https://www.amzaon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7",
                "text": AMAZON_RETURN_POLICY_TEXT,
            },
            "Related policies": [],
        },
        "persona": persona,
        # Short agent-visible brief (no conditions or return reasons)
        "detail_agent": detail_agent,
        "task": {
            # --- Backward-compat flat fields (many consumers read these directly) ---
            "order_id": order_id,
            "order_date": purchase_date,
            "purchase_date": purchase_date,
            "delivery_date": delivery_date,
            "products_involved": products_involved,
            "items": selected_items,
            "return_reasons": dict(zip(
                [item["product_name"] for item in selected_items],
                return_reasons,
            )),
            # --- Three-part structure ---
            "basic_info": basic_info,
            "return_details": return_details,
            "customer_behavior": customer_behavior,
            # --- Ground truth for verify_return (basic facts + return narrative) ---
            "detail": combined_detail,
            "task": combined_detail,
            # --- Customer LLM only: brief of openly presented intent ---
            "customer_agent_info": (task_detail or {}).get("customer_agent_info", ""),
            # --- Internal-only ---
            "related_policy_issues": policy_issues,
            "complexity_level": (task_detail or {}).get("complexity_level", "High Complexity"),
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

_PLANNING_PATTERNS = [
    "please hold on",
    "i'll look into",
    "i'll check this",
    "i'll check the",
    "i'll review",
    "i'll update you",
    "i'll get back",
    "i need a moment",
    "one moment please",
    "give me a moment",
    "i'll be right back",
    "i'll gather",
    "i'll retrieve",
    "i'll pull up",
    "i'll investigate",
    "let me look into",
    "i'll look up",
    "shortly with next steps",
    "get back to you",
    "next steps shortly",
    "confirm the return eligibility and process",
    "i'll confirm",
    "i'll verify",
    "i'll process",
]


def _has_planning_phrases(message: str) -> bool:
    """Return True if the message contains async-promise planning phrases."""
    msg_lower = message.lower()
    return any(p in msg_lower for p in _PLANNING_PATTERNS)


def _rewrite_as_direct_message(state: ConversationState, provider: LLMProvider) -> str:
    """Make a focused LLM call to produce a direct, non-planning agent message."""
    history_str = state.get_formatted_history_str()
    messages = [
        {
            "role": "system",
            "content": (
                "You are an Amzaon customer support agent. "
                "FORBIDDEN phrases — NEVER use: 'please hold on', 'I'll check', 'I'll review', "
                "'I'll update you shortly', 'I'll get back to you', 'one moment', 'give me a moment'. "
                "Write a direct response using ONLY facts from tool results already in the conversation history. "
                "End with a specific question the customer can answer immediately."
            ),
        },
        {
            "role": "user",
            "content": (
                "Conversation history (tool results are included):\n"
                f"{history_str}\n\n"
                "Write the agent's next customer-facing message. "
                "No planning phrases. Must end with a concrete question:"
            ),
        },
    ]
    try:
        resp = provider.call_text_only(messages=messages, temperature=0.5, max_tokens=300)
        return (resp.content or "").strip()
    except Exception:
        return ""


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

    # Phase 1: Tool-execution loop — keep calling until no new tools are needed.
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
            continue
        break

    # Phase 2: Message quality check — if the agent wrote a planning message,
    # make a focused rewrite call with hard constraints against planning phrases.
    message = agent_resp.message or ""
    if not message or _has_planning_phrases(message):
        rewritten = _rewrite_as_direct_message(state, provider)
        if rewritten and not _has_planning_phrases(rewritten):
            message = rewritten

    if not message:
        message = "Could you tell me more about the issue with your item? For example, what condition is it in and when did it arrive?"

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
        "Hello! Welcome to Amzaon Customer Service. How can I assist you today? "
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
    """Return a ≤30-word hint telling the player what to say next, using LLM."""
    last_agent_message = ""
    for turn in reversed(state.history):
        if turn.turn == "agent" and turn.message:
            last_agent_message = turn.message
            break

    task = scenario.get("task", {})
    order_id = task.get("order_id", "")
    items = task.get("items", [])
    product_names = ", ".join(
        i.get("product_name", "") if isinstance(i, dict) else str(i) for i in items
    ) or "the item"
    return_reasons = task.get("return_reasons", {})
    if isinstance(return_reasons, dict):
        reasons_text = "; ".join(f"{k}: {v}" for k, v in return_reasons.items())
    else:
        reasons_text = str(return_reasons)

    task_description = task.get("scenario_description") or task.get("description") or (
        f"Return {product_names}"
        + (f" (order {order_id})" if order_id else "")
        + (f" — reason: {reasons_text}" if reasons_text else "")
    )

    sentences = last_agent_message.replace("?", "?\n").split("\n")
    last_question = next((s.strip() for s in reversed(sentences) if "?" in s and s.strip()), "")
    agent_prompt_text = last_question or last_agent_message[:200]

    messages = [
        {"role": "system", "content": "You are a hint generator for a customer support negotiation game. Output only the hint, no preamble."},
        {"role": "user", "content": (
            f"Task: {task_description}\n\n"
            f"Agent just asked: {agent_prompt_text}\n\n"
            "Write a hint (≤30 words) telling the player exactly what to say next "
            "to advance their return request. Be specific, use details from the task."
        )},
    ]
    resp = provider.call_text_only(messages=messages)
    hint = (resp.content or "").strip()
    if not hint:
        return f"Answer the agent's question using your task details: {task_description[:80]}."
    return hint


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

    return_details = task.get("return_details", "")
    complexity = task.get("complexity_level", "")

    if return_details:
        complexity_html = ""
        if complexity:
            complexity_html = (
                f'<div style="margin-top:6px;font-size:0.78rem;color:#bf360c;font-weight:600;">'
                f'Difficulty: {complexity}</div>'
            )
        return (
            f"<strong>You are {name}{age_loc}.</strong><br><br>"
            f"{return_details}"
            + complexity_html
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
        f"You recently purchased {item_name} from Amzaon, "
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
        f"Hi, I need help returning something I recently purchased — the {item_name}.",
        (
            f"Hello, I'm {name}. I'd like to start a return — "
            f"the {item_name} {reason.lower()}. Can you help me with this?"
        ),
        (
            f"Good day. I'm reaching out about a return. "
            f"I recently bought the {item_name} and I'm having an issue with it."
        ),
    ]
