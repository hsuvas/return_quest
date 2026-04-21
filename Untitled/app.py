"""
app.py — Return Quest (Gamified Edition)
=========================================


Run from project root:
     streamlit run app.py
"""

import os
import sys
import json
import uuid
import datetime
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# Path bootstrap (must happen before any local imports)
# ---------------------------------------------------------------------------

_SHOWCASE_DIR = Path(__file__).resolve().parent

if str(_SHOWCASE_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOWCASE_DIR))

try:
    from dotenv import load_dotenv
    for _env_path in (_SHOWCASE_DIR / ".env", _SHOWCASE_DIR.parent / ".env"):
        if _env_path.exists():
            load_dotenv(_env_path)
            break
except ImportError:
    pass

import threading
import time

import httpx
import streamlit as st
import streamlit.components.v1 as components
import uvicorn

from showcase_backend import (
    load_products_balanced,
    load_personas,
    get_product_categories,
    build_scenario,
    make_provider,
    generate_task_detail,
    generate_narrative,
    generate_starters,
    RESOLUTION_DISPLAY,
)

# ---------------------------------------------------------------------------
# API server (FastAPI) — started once per Streamlit worker process
# ---------------------------------------------------------------------------

_API_PORT = 8765
_API_BASE = f"http://127.0.0.1:{_API_PORT}"


def _start_api_server():
    import server as _srv  # local import so path bootstrap has run by now
    uvicorn.run(_srv.app, host="127.0.0.1", port=_API_PORT, log_level="warning")


if "api_server_started" not in st.session_state:
    t = threading.Thread(target=_start_api_server, daemon=True)
    t.start()
    # Wait until the server is actually accepting connections (up to 10s)
    for _ in range(20):
        try:
            httpx.get(f"{_API_BASE}/api/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.5)
    st.session_state["api_server_started"] = True

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Return Quest",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# PIXEL ART SPRITE SYSTEM
# ===========================================================================

def _make_person(skin, hair, shirt, pants, shoes, eyes="#2c2c2c", detail=None):
    """
    Generate a person pixel sprite as a list of (col, row, hex_color).
    Grid: 10 cols (0–9) × 18 rows (0–17).
    detail: optional dict {(col,row): color} for extra pixels.
    """
    pixels = {}

    def put(col, row, color):
        pixels[(col, row)] = color

    # Hair — rows 0–1
    for c in range(3, 7):
        put(c, 0, hair)
    for c in range(2, 8):
        put(c, 1, hair)

    # Face — rows 2–5
    for r in range(2, 6):
        for c in range(2, 8):
            put(c, r, skin)

    # Eyes — row 3
    put(3, 3, eyes)
    put(6, 3, eyes)

    # Neck — row 6
    put(4, 6, skin)
    put(5, 6, skin)

    # Body / shirt — rows 7–11
    for r in range(7, 12):
        for c in range(2, 8):
            put(c, r, shirt)

    # Arms — rows 8–10, cols 1 and 8
    for r in range(8, 11):
        put(1, r, shirt)
        put(8, r, shirt)

    # Pants — rows 12–15, split legs
    for r in range(12, 16):
        for c in range(2, 5):
            put(c, r, pants)
        for c in range(5, 8):
            put(c, r, pants)

    # Shoes — row 17
    for c in range(2, 5):
        put(c, 17, shoes)
    for c in range(5, 8):
        put(c, 17, shoes)

    # Extra detail pixels
    if detail:
        for (c, r), color in detail.items():
            put(c, r, color)

    return [(c, r, color) for (c, r), color in pixels.items()]


def _make_robot(body, visor, accent, antenna_color=None):
    """
    Generate a robot pixel sprite as a list of (col, row, hex_color).
    Grid: 10 cols (0–9) × 15 rows (0–14).
    """
    pixels = {}

    def put(col, row, color):
        pixels[(col, row)] = color

    # Antenna — rows 0–1 (optional)
    if antenna_color:
        put(4, 0, antenna_color)
        put(5, 0, antenna_color)
        put(4, 1, antenna_color)
        put(5, 1, antenna_color)

    # Head — rows 2–6
    for r in range(2, 7):
        for c in range(2, 8):
            put(c, r, body)

    # Visor — rows 3–4, cols 3–6
    for r in range(3, 5):
        for c in range(3, 7):
            put(c, r, visor)

    # Body — rows 7–11
    for r in range(7, 12):
        for c in range(2, 8):
            put(c, r, body)

    # Accent stripe on chest — row 8
    for c in range(3, 7):
        put(c, 8, accent)

    # Arms — rows 7–10
    for r in range(7, 11):
        put(1, r, body)
        put(8, r, body)

    # Legs — rows 12–14
    for r in range(12, 15):
        put(3, r, body)
        put(4, r, body)
        put(5, r, body)
        put(6, r, body)

    return [(c, r, color) for (c, r), color in pixels.items()]


def pixel_sprite_html(pixels, px=6, grid_w=10, grid_h=18):
    """
    Render a pixel sprite as an HTML table of colored cells.
    This is the most reliable approach in Streamlit's HTML sandbox.
    """
    grid = {(c, r): color for c, r, color in pixels}
    rows_html = []
    for r in range(grid_h):
        cells = []
        for c in range(grid_w):
            color = grid.get((c, r))
            bg = f"background:{color};" if color else "background:transparent;"
            cells.append(
                f'<td style="width:{px}px;height:{px}px;{bg}padding:0;margin:0;border:none;"></td>'
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    total_w = grid_w * px
    total_h = grid_h * px
    return (
        f'<div style="display:flex;justify-content:center;margin:6px auto;">'
        f'<table style="border-collapse:collapse;width:{total_w}px;height:{total_h}px;'
        f'table-layout:fixed;border-spacing:0;">'
        + "".join(rows_html)
        + "</table></div>"
    )


# ---------------------------------------------------------------------------
# Sprite definitions
# ---------------------------------------------------------------------------

PERSONA_SPRITES = {
    # Professional male — red tie at center chest
    "professional_m": _make_person(
        skin="#f5c5a3", hair="#2c1810", shirt="#1e3a5f",
        pants="#263238", shoes="#1a1a1a",
        detail={
            (4, 6): "#ef5350", (5, 6): "#ef5350",   # tie knot at neck
            (4, 7): "#c62828", (5, 7): "#c62828",   # tie body
            (4, 8): "#c62828", (5, 8): "#c62828",
            (4, 9): "#b71c1c", (5, 9): "#b71c1c",
            (4, 10): "#b71c1c", (5, 10): "#b71c1c",
        },
    ),
    # Professional female — long side hair falling past shoulders
    "professional_f": _make_person(
        skin="#f5c5a3", hair="#8b4513", shirt="#c2185b",
        pants="#4a148c", shoes="#1a1a1a",
        detail={
            (1, 4): "#8b4513", (1, 5): "#8b4513", (1, 6): "#8b4513",
            (1, 7): "#8b4513", (1, 8): "#8b4513", (1, 9): "#8b4513",
            (8, 4): "#8b4513", (8, 5): "#8b4513", (8, 6): "#8b4513",
            (8, 7): "#8b4513", (8, 8): "#8b4513", (8, 9): "#8b4513",
        },
    ),
    # Creative male — bright scarf at neck
    "creative_m": _make_person(
        skin="#d4956a", hair="#ff6b35", shirt="#ff9800",
        pants="#7b1fa2", shoes="#e91e63",
        detail={
            (2, 6): "#ff5722", (3, 6): "#ff5722", (4, 6): "#ff5722",
            (5, 6): "#ff5722", (6, 6): "#ff5722", (7, 6): "#ff5722",
        },
    ),
    # Creative female — long pink side hair
    "creative_f": _make_person(
        skin="#d4956a", hair="#e91e63", shirt="#9c27b0",
        pants="#f06292", shoes="#7b1fa2",
        detail={
            (1, 4): "#e91e63", (1, 5): "#e91e63", (1, 6): "#e91e63",
            (1, 7): "#e91e63", (1, 8): "#e91e63", (1, 9): "#e91e63",
            (8, 4): "#e91e63", (8, 5): "#e91e63", (8, 6): "#e91e63",
            (8, 7): "#e91e63", (8, 8): "#e91e63", (8, 9): "#e91e63",
        },
    ),
    # Academic male — glasses frames around eyes
    "academic_m": _make_person(
        skin="#f5c5a3", hair="#5d4037", shirt="#558b2f",
        pants="#37474f", shoes="#212121",
        detail={
            # Glasses frame (eye positions are col 3 and col 6, row 3)
            (2, 3): "#6d4c41", (4, 3): "#6d4c41",   # left lens sides
            (2, 4): "#6d4c41", (3, 4): "#6d4c41", (4, 4): "#6d4c41",  # left lens bottom
            (5, 3): "#6d4c41", (7, 3): "#6d4c41",   # right lens sides
            (5, 4): "#6d4c41", (6, 4): "#6d4c41", (7, 4): "#6d4c41", # right lens bottom
            # Book in hand
            (8, 9): "#fff8e1", (9, 9): "#fff8e1",
            (8, 10): "#fff8e1", (9, 10): "#fff8e1",
        },
    ),
    # Academic female — side bun hairstyle
    "academic_f": _make_person(
        skin="#fce4d6", hair="#4e342e", shirt="#1565c0",
        pants="#37474f", shoes="#212121",
        detail={
            (7, 0): "#4e342e", (8, 0): "#4e342e",   # bun top
            (8, 1): "#4e342e", (9, 1): "#4e342e",   # bun side
            (8, 2): "#4e342e", (9, 2): "#4e342e",   # bun base
        },
    ),
    # Youth male — dark cap covering hair
    "youth_m": _make_person(
        skin="#f5c5a3", hair="#ffeb3b", shirt="#00bcd4",
        pants="#f44336", shoes="#212121",
        detail={
            # Cap overrides hair rows 0–1, full width
            (1, 0): "#263238", (2, 0): "#263238", (3, 0): "#263238",
            (4, 0): "#263238", (5, 0): "#263238", (6, 0): "#263238",
            (7, 0): "#263238", (8, 0): "#263238",
            (1, 1): "#263238", (2, 1): "#263238", (3, 1): "#263238",
            (4, 1): "#263238", (5, 1): "#263238", (6, 1): "#263238",
            (7, 1): "#263238", (8, 1): "#263238",
            # Cap brim extends wider at row 2
            (1, 2): "#1a1a1a", (8, 2): "#1a1a1a",
        },
    ),
    # Youth female — side ponytail
    "youth_f": _make_person(
        skin="#fce4d6", hair="#f48fb1", shirt="#ff4081",
        pants="#7e57c2", shoes="#212121",
        detail={
            (8, 2): "#f48fb1", (8, 3): "#f48fb1", (8, 4): "#f48fb1",
            (8, 5): "#f48fb1", (8, 6): "#f48fb1", (8, 7): "#f48fb1",
            (9, 3): "#f48fb1", (9, 4): "#f48fb1", (9, 5): "#f48fb1", (9, 6): "#f48fb1",
        },
    ),
    "default_m": _make_person(
        skin="#f5c5a3", hair="#4a3728", shirt="#2196f3",
        pants="#37474f", shoes="#1a1a1a",
    ),
    "default_f": _make_person(
        skin="#fce4d6", hair="#c0392b", shirt="#e91e63",
        pants="#37474f", shoes="#1a1a1a",
    ),
}

# ---------------------------------------------------------------------------
# Character image mapping (preset → filename, custom → name-hash pick)
# Images are embedded as base64 data URLs to avoid static-serving URL issues.
# ---------------------------------------------------------------------------

_CHAR_IMG_DIR = _SHOWCASE_DIR / "images" / "character"

_PERSONA_IMAGE_MAP = {
    "Marco_Technology_Italy_01":   "marco_deluca",
    "Catherine_History_Ireland_02": "catherine_oneil",
    "Eratios_Academia_Greece_03":  "eratikos_nikolaou",
    "Samantha_Education_UK_04":    "samantha_lewis",
    "David_Engineering_USA_05":    "david_morales",
}
_IMAGE_POOL = [
    "marco_deluca",
    "catherine_oneil",
    "eratikos_nikolaou",
    "samantha_lewis",
    "david_morales",
]


@st.cache_data(show_spinner=False)
def _load_char_images() -> dict:
    """Load all character PNGs as base64 data URLs (cached once at startup)."""
    import base64
    result = {}
    for fname in _IMAGE_POOL:
        path = _CHAR_IMG_DIR / f"{fname}.png"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            result[fname] = f"data:image/png;base64,{data}"
    return result


def _get_persona_data_url(persona: dict) -> str:
    imgs = _load_char_images()
    pid = persona.get("Persona_id", "")
    if pid in _PERSONA_IMAGE_MAP:
        fname = _PERSONA_IMAGE_MAP[pid]
    else:
        name = persona.get("Name", "custom")
        idx = sum(ord(c) for c in name) % len(_IMAGE_POOL)
        fname = _IMAGE_POOL[idx]
    return imgs.get(fname, "")


def persona_img_html(persona: dict, height: int = 130) -> str:
    """Return an <img> tag for this persona's character image."""
    url = _get_persona_data_url(persona)
    if not url:
        return ""
    return (
        f'<div style="text-align:center;margin:4px 0 6px;">'
        f'<img src="{url}" style="height:{height}px;width:auto;'
        f'max-width:100%;object-fit:contain;border-radius:4px;">'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Agent images
# ---------------------------------------------------------------------------

_AGENT_IMG_DIR = _SHOWCASE_DIR / "images" / "agent"

_AGENT_IMAGE_MAP = {
    "DIRECT":      "direct_agent",
    "FAIR":        "fair_agent",
    "AGREEABLE":   "agreeable",
    "HELPFUL":     "helpful_agent",
    "VERY_HELPFUL": "very_helpful",
}


@st.cache_data(show_spinner=False)
def _load_agent_images() -> dict:
    """Load all agent PNGs as base64 data URLs (cached once at startup)."""
    import base64
    result = {}
    for fname in _AGENT_IMAGE_MAP.values():
        path = _AGENT_IMG_DIR / f"{fname}.png"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            result[fname] = f"data:image/png;base64,{data}"
    return result


def agent_img_html(agent_key: str, height: int = 130) -> str:
    """Return an <img> tag for an agent image by key (DIRECT, FAIR, etc.)."""
    fname = _AGENT_IMAGE_MAP.get(agent_key, "")
    url = _load_agent_images().get(fname, "")
    if not url:
        return ""
    return (
        f'<div style="text-align:center;margin:4px 0 6px;">'
        f'<img src="{url}" style="height:{height}px;width:auto;'
        f'max-width:100%;object-fit:contain;border-radius:4px;">'
        f'</div>'
    )


AGENT_SPRITES = {
    "DIRECT": _make_robot(
        body="#546e7a", visor="#ef5350", accent="#b71c1c",
    ),
    "FAIR": _make_robot(
        body="#78909c", visor="#b0bec5", accent="#455a64",
    ),
    "AGREEABLE": _make_robot(
        body="#388e3c", visor="#81c784", accent="#1b5e20",
        antenna_color="#ffeb3b",
    ),
    "HELPFUL": _make_robot(
        body="#1565c0", visor="#42a5f5", accent="#0d47a1",
        antenna_color="#ffd54f",
    ),
    "VERY_HELPFUL": _make_robot(
        body="#6a1b9a", visor="#ce93d8", accent="#ffd700",
        antenna_color="#ffd700",
    ),
}


def get_persona_sprite_key(persona):
    """Map persona attributes to a sprite key."""
    age = persona.get("Age-range", "30-35")
    sector = persona.get("Job Sector", "")
    gender = persona.get("Gender", "")
    is_female = "female" in gender.lower() or "woman" in gender.lower()
    suffix = "_f" if is_female else "_m"

    young = any(x in age for x in ["18", "20", "22", "25"])
    creative = any(s in sector for s in ["Art", "Media", "Music", "Design", "Film"])
    academic = any(s in sector for s in ["Academia", "Education", "Research", "Science"])
    professional = any(s in sector for s in ["Tech", "Finance", "Law", "Engineer", "Business"])

    if young:
        return f"youth{suffix}"
    if creative:
        return f"creative{suffix}"
    if academic:
        return f"academic{suffix}"
    if professional:
        return f"professional{suffix}"
    return f"default{suffix}"


# ===========================================================================
# LANGUAGE STRINGS
# ===========================================================================

STRINGS = {
    "en": {
        "title": "Return Quest",
        "lang_toggle": "DE",
        "step_labels": [
            "1 · Your Avatar",
            "2 · Pick Items",
            "3 · Choose Agent",
            "4 · Your Mission",
            "5 · Chat",
            "6 · Outcome",
        ],
        "restart": "Restart Game",
        "kid_mode_on": "🧒 Kid Mode: ON",
        "kid_mode_off": "🎮 Kid Mode: OFF",
        # Step 1
        "step1_title": "Choose Your Character",
        "step1_sub": "Pick a persona to play as in this return scenario.",
        "preset_avatars": "Preset Characters",
        "create_own": "Create My Own Character",
        "custom_expand": "Expand to build a custom character",
        "next_items": "Next: Pick Items",
        "select": "Select",
        "selected": "Selected",
        "please_select_persona": "Please select or create a character to continue.",
        "avatar_selected": "Character selected",
        "personality": "Personality",
        # Step 2
        "step2_title": "Select Items to Return",
        "step2_sub": "Choose 1–3 products to return.",
        "search_placeholder": "Search products...",
        "category": "Category",
        "all_categories": "All Categories",
        "showing": "Showing",
        "of": "of",
        "products": "products",
        "return_reason": "Return reason",
        "selected_items": "Selected Items",
        "select_at_least_one": "Select at least one item to continue.",
        "next_agent": "Next: Choose Agent",
        "amazon": "Amazon",
        "third_party": "3rd Party",
        "items_counter": "items selected",
        # Step 3
        "step3_title": "Choose Your Agent",
        "step3_sub": "Pick the personality powering the support agent.",
        "agent_personality": "Agent Personality",
        "model_label": "Model",
        "next_mission": "Next: Your Mission",
        # Step 4
        "step4_title": "Your Mission",
        "step4_sub": "Read your scenario, then choose how to start the conversation.",
        "how_to_start": "How would you like to open the conversation?",
        "write_own": "Write my own message...",
        "begin_chat": "Begin Conversation",
        "generating": "Generating your mission...",
        "write_msg_first": "Write or choose a message to start.",
        # Step 5
        "step5_title": "Chat",
        "helper_title": "Helper",
        "your_character": "Your character",
        "turn": "Turn",
        "get_hint": "Get a Hint",
        "use_hint": "Use this hint",
        "type_reply": "Type your reply...",
        "send": "Send",
        "hint_generating": "Generating hint...",
        "turn_limit_warning": "Approaching max turns — the agent will need to conclude soon.",
        "view_resolution": "View Outcome",
        "agent_thinking": "Agent is thinking...",
        # Step 6
        "step6_title": "Outcome",
        "accepted": "Outcome Accepted!",
        "no_resolution": "No resolution was reached.",
        "back_to_chat": "Back to Chat",
        "accept": "Accept Outcome",
        "reject": "Not satisfied — Continue",
        "play_again": "Play Again",
        "turns": "Turns",
        "tool_calls": "Agent Tool Calls",
        "policy_tensions": "Policy Tensions",
        "return_score": "Return Score",
        "resolution_details": "Resolution Details",
        "next_steps": "Your Next Steps",
        "conditions": "Conditions",
        "full_transcript": "Full Conversation Transcript",
    },
    "de": {
        "title": "Rückgabe-Quest",
        "lang_toggle": "EN",
        "step_labels": [
            "1 · Dein Avatar",
            "2 · Artikel wählen",
            "3 · Agent wählen",
            "4 · Deine Mission",
            "5 · Chat",
            "6 · Ergebnis",
        ],
        "restart": "Neu starten",
        "kid_mode_on": "🧒 Kinder-Modus: AN",
        "kid_mode_off": "🎮 Kinder-Modus: AUS",
        # Step 1
        "step1_title": "Wähle deinen Charakter",
        "step1_sub": "Wähle eine Persona für dieses Rückgabe-Szenario.",
        "preset_avatars": "Voreingestellte Charaktere",
        "create_own": "Eigenen Charakter erstellen",
        "custom_expand": "Aufklappen zum Erstellen",
        "next_items": "Weiter: Artikel wählen",
        "select": "Wählen",
        "selected": "Gewählt",
        "please_select_persona": "Bitte wähle oder erstelle einen Charakter.",
        "avatar_selected": "Charakter gewählt",
        "personality": "Persönlichkeit",
        # Step 2
        "step2_title": "Artikel zum Zurückgeben",
        "step2_sub": "Wähle 1–3 Produkte aus.",
        "search_placeholder": "Produkte suchen...",
        "category": "Kategorie",
        "all_categories": "Alle Kategorien",
        "showing": "Zeige",
        "of": "von",
        "products": "Produkte",
        "return_reason": "Rückgabegrund",
        "selected_items": "Ausgewählte Artikel",
        "select_at_least_one": "Bitte wähle mindestens einen Artikel.",
        "next_agent": "Weiter: Agent wählen",
        "amazon": "Amazon",
        "third_party": "Drittanbieter",
        "items_counter": "Artikel ausgewählt",
        # Step 3
        "step3_title": "Wähle deinen Agenten",
        "step3_sub": "Wähle die Persönlichkeit des Support-Agenten.",
        "agent_personality": "Agenten-Persönlichkeit",
        "model_label": "Modell",
        "next_mission": "Weiter: Deine Mission",
        # Step 4
        "step4_title": "Deine Mission",
        "step4_sub": "Lies dein Szenario und wähle, wie du anfangen möchtest.",
        "how_to_start": "Wie möchtest du das Gespräch beginnen?",
        "write_own": "Eigene Nachricht schreiben...",
        "begin_chat": "Gespräch beginnen",
        "generating": "Mission wird generiert...",
        "write_msg_first": "Schreibe oder wähle eine Nachricht.",
        # Step 5
        "step5_title": "Chat",
        "helper_title": "Helfer",
        "your_character": "Dein Charakter",
        "turn": "Runde",
        "get_hint": "Hinweis holen",
        "use_hint": "Hinweis verwenden",
        "type_reply": "Antwort eingeben...",
        "send": "Senden",
        "hint_generating": "Hinweis wird generiert...",
        "turn_limit_warning": "Fast am Limit — der Agent muss bald abschließen.",
        "view_resolution": "Ergebnis anzeigen",
        "agent_thinking": "Agent denkt nach...",
        # Step 6
        "step6_title": "Ergebnis",
        "accepted": "Ergebnis akzeptiert!",
        "no_resolution": "Kein Ergebnis wurde erzielt.",
        "back_to_chat": "Zurück zum Chat",
        "accept": "Ergebnis akzeptieren",
        "reject": "Nicht zufrieden — Weiter",
        "play_again": "Nochmal spielen",
        "turns": "Runden",
        "tool_calls": "Tool-Aufrufe",
        "policy_tensions": "Regelspannungen",
        "return_score": "Rückgabe-Score",
        "resolution_details": "Ergebnis-Details",
        "next_steps": "Nächste Schritte",
        "conditions": "Bedingungen",
        "full_transcript": "Vollständiges Gespräch",
    },
}

RETURN_REASONS = {
    "en": ["Defective", "Wrong Item", "Changed Mind", "Damaged in Shipping", "Other"],
    "de": ["Defekt", "Falscher Artikel", "Meinung geändert", "Transportschaden", "Sonstiges"],
}

# Map DE reason back to EN for backend compatibility
_REASON_DE_TO_EN = {
    "Defekt": "Defective",
    "Falscher Artikel": "Wrong Item",
    "Meinung geändert": "Changed Mind",
    "Transportschaden": "Damaged in Shipping",
    "Sonstiges": "Other",
}


def t(key):
    lang = st.session_state.get("lang", "en")
    return STRINGS.get(lang, STRINGS["en"]).get(key, key)


def get_reasons():
    return RETURN_REASONS.get(st.session_state.get("lang", "en"), RETURN_REASONS["en"])


def reason_to_en(reason):
    """Normalize a potentially German reason label to English for backend."""
    return _REASON_DE_TO_EN.get(reason, reason)


# Kid-mode simplified trait labels
TRAIT_NAMES_KID = {
    "en": {
        "Agreeableness": "Friendliness",
        "Neuroticism": "Calmness",
        "Openness": "Curiosity",
        "Conscientiousness": "Focus",
        "Extraversion": "Energy",
    },
    "de": {
        "Agreeableness": "Freundlichkeit",
        "Neuroticism": "Gelassenheit",
        "Openness": "Neugier",
        "Conscientiousness": "Fokus",
        "Extraversion": "Energie",
    },
}


def trait_name(key):
    """Return display name for a personality trait; simplified in kid mode."""
    if st.session_state.get("kid_mode", False):
        lang = st.session_state.get("lang", "en")
        return TRAIT_NAMES_KID.get(lang, TRAIT_NAMES_KID["en"]).get(key, key)
    return key


# ===========================================================================
# CSS — Pixel-game theme
# Injected via JS into parent document head to avoid Streamlit's style-tag
# stripping (which renders CSS as visible text even with unsafe_allow_html).
# ===========================================================================

def _build_css():
    """Return theme CSS string. Switches between dark (default) and light (kid mode)."""
    kid = st.session_state.get("kid_mode", False)
    if kid:
        heading_color = "#ff6b00"
        heading_shadow = "2px 2px 0 #ffd1b3"
        sub_color = "#5c007a"
        name_color = "#1a1a2e"
        card_sub_color = "#616161"
        trait_color = "#7b1fa2"
        product_name_color = "#1a1a2e"
        product_sub_color = "#616161"
        narrative_bg = "#fff3e0"
        narrative_border = "#ff6b00"
        narrative_color = "#1a1a2e"
        narrative_title_color = "#ff6b00"
        agent_bubble_bg = "#2e7d32"
        agent_bubble_color = "#ffffff"
        tool_chip_bg = "#fff3e0"
        tool_chip_border = "#ff6b00"
        tool_chip_color = "#e65100"
        bar_color = "#ff6b00"
        sidebar_bg = "#fce4ec"
        sidebar_text = "#1a1a2e"
        sidebar_btn_border = "#ef9a9a"
        sidebar_btn_hover = "#f8bbd0"
    else:
        # Clean light teal theme — easy to read for all ages
        heading_color = "#00695c"
        heading_shadow = "2px 2px 0 #b2dfdb"
        sub_color = "#37474f"
        name_color = "#1a1a2e"
        card_sub_color = "#546e7a"
        trait_color = "#00695c"
        product_name_color = "#1a1a2e"
        product_sub_color = "#546e7a"
        narrative_bg = "#e0f2f1"
        narrative_border = "#00695c"
        narrative_color = "#1a1a2e"
        narrative_title_color = "#00695c"
        agent_bubble_bg = "#1b5e20"
        agent_bubble_color = "#e8f5e9"
        tool_chip_bg = "#f5f5f5"
        tool_chip_border = "#00695c"
        tool_chip_color = "#00695c"
        bar_color = "#00695c"
        sidebar_bg = "#eceff1"
        sidebar_text = "#263238"
        sidebar_btn_border = "#90a4ae"
        sidebar_btn_hover = "#e0e0e0"

    return f"""
html, body, [class*="css"] {{ font-family: 'Segoe UI', Arial, sans-serif; }}

.px-heading {{
    font-family: 'Press Start 2P', monospace;
    font-size: 1.1rem;
    color: {heading_color};
    text-shadow: {heading_shadow};
    margin-bottom: 0.3rem;
    line-height: 1.6;
}}
.px-sub {{
    color: {sub_color};
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}}

.px-card-name {{ font-weight: 700; font-size: 1.05rem; margin: 6px 0 2px; color: {name_color}; }}
.px-card-sub  {{ font-size: 0.78rem; color: {card_sub_color}; margin: 2px 0; }}
.px-card-trait {{
    font-size: 0.72rem; color: {trait_color}; margin: 2px 0;
    font-family: 'Courier New', monospace; letter-spacing: 2px;
}}

.px-card-wrap {{ position: relative; }}
.px-card-tooltip {{
    display: none;
    position: absolute;
    bottom: 105%; left: 50%; transform: translateX(-50%);
    background: #263238; color: #eceff1;
    font-size: 0.78rem; line-height: 1.45;
    padding: 8px 12px; border-radius: 6px;
    width: 220px; text-align: left;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 999; pointer-events: none;
}}
.px-card-tooltip::after {{
    content: ''; position: absolute;
    top: 100%; left: 50%; transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: #263238;
}}
.px-card-wrap:hover .px-card-tooltip {{ display: block; }}

.px-product-name {{ font-weight: 700; font-size: 0.88rem; color: {product_name_color}; margin: 4px 0 2px; }}
.px-product-sub  {{ font-size: 0.75rem; color: {product_sub_color}; }}
.px-badge {{
    display: inline-block; padding: 2px 7px; border-radius: 3px;
    font-size: 0.7rem; font-weight: 600; margin: 2px 2px 0;
}}
.px-badge-amz {{ background: #e65100; color: #fff; }}
.px-badge-3p  {{ background: #1565c0; color: #fff; }}

.px-narrative {{
    border: 4px solid {narrative_border}; border-radius: 4px; padding: 18px 20px;
    background: {narrative_bg}; color: {narrative_color}; font-size: 0.92rem;
    line-height: 1.7; margin-bottom: 18px;
}}
.px-narrative-title {{
    font-family: 'Press Start 2P', monospace; font-size: 0.75rem;
    color: {narrative_title_color}; margin-bottom: 10px; letter-spacing: 2px;
}}

.customer-bubble {{
    background: #1565c0; border-radius: 14px 14px 4px 14px;
    padding: 10px 14px; margin: 6px 0 6px 60px;
    max-width: 72%; float: right; clear: both; color: #fff; font-size: 0.92rem;
}}
.agent-bubble {{
    background: {agent_bubble_bg}; border-radius: 14px 14px 14px 4px;
    padding: 10px 14px; margin: 6px 60px 6px 0;
    max-width: 72%; float: left; clear: both; color: {agent_bubble_color}; font-size: 0.92rem;
}}
.tool-chip {{
    background: {tool_chip_bg}; border: 1px solid {tool_chip_border}; border-radius: 3px;
    padding: 2px 8px; font-size: 0.72rem; color: {tool_chip_color};
    margin: 2px 3px; display: inline-block; font-family: monospace;
}}
.clearfix {{ clear: both; }}

.px-bar {{ font-family: 'Courier New', monospace; color: {bar_color}; font-size: 0.9rem; letter-spacing: 3px; }}

section[data-testid="stSidebar"] {{ background: {sidebar_bg} !important; }}
section[data-testid="stSidebar"] * {{ color: {sidebar_text} !important; }}

section[data-testid="stSidebar"] .stButton > button {{
    color: {sidebar_text} !important;
    background: transparent !important;
    border: 1px solid {sidebar_btn_border} !important;
    border-radius: 4px !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {sidebar_btn_hover} !important;
}}
"""


def _inject_css():
    """Inject (or replace) theme CSS into the parent document."""
    _css_json = json.dumps(_build_css())
    components.html(
        f"""
        <script>
        (function() {{
            var p = window.parent.document;

            // Google Font (once)
            if (!p.querySelector('link[data-px-font]')) {{
                var lnk = p.createElement('link');
                lnk.rel = 'stylesheet';
                lnk.href = 'https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap';
                lnk.setAttribute('data-px-font', '1');
                p.head.appendChild(lnk);
            }}

            // Always replace CSS so theme switches take effect immediately
            var old = p.querySelector('style[data-px-style]');
            if (old) {{ old.remove(); }}
            var st = p.createElement('style');
            st.setAttribute('data-px-style', '1');
            st.textContent = {_css_json};
            p.head.appendChild(st);
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )

# ===========================================================================
# Session state
# ===========================================================================

def init_state():
    defaults = {
        "lang": "en",
        "kid_mode": False,
        "session_id": str(uuid.uuid4()),
        "step": 1,
        "persona": None,
        "custom_persona_mode": False,
        "selected_items": [],
        "return_reasons": [],
        "agent_persona_choice": "FAIR",
        "model_choice": "gpt-4.1-2025-04-14",
        "scenario": None,
        "narrative": "",
        "starters": [],
        "first_message": "",
        "api_session_id": None,    # server-side session ID
        "tool_calls_count": 0,     # cumulative agent tool calls (from server)
        "messages": [],
        "resolution": None,
        "accepted_resolution": False,
        "turn_count": 0,
        "finished": False,
        "hint_text": None,
        "hint_used_this_turn": False,
        "turn_hint_flags": [],
        "verification_result": None,
        "input_counter": 0,
        "_prefill_msg": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# Inject theme CSS (after init_state so kid_mode session state is available)
_inject_css()


def get_theme():
    """Return inline color dict for cards/chips based on current mode."""
    if st.session_state.get("kid_mode", False):
        return {
            "card_bg": "#ffffff",
            "card_border": "#e0e0e0",
            "card_bg_sel": "#fff3e0",
            "card_border_sel": "#ff6b00",
            "card_text": "#1a1a2e",
            "card_sub": "#757575",
            "chip_bg": "#f5f5f5",
            "chip_border": "#bdbdbd",
            "chip_color": "#424242",
            "product_bg": "#ffffff",
            "product_bg_sel": "#fff3e0",
            "product_border": "#e0e0e0",
            "product_border_sel": "#ff6b00",
        }
    else:
        return {
            "card_bg": "#ffffff",
            "card_border": "#cfd8dc",
            "card_bg_sel": "#e0f2f1",
            "card_border_sel": "#00695c",
            "card_text": "#1a1a2e",
            "card_sub": "#546e7a",
            "chip_bg": "#f5f5f5",
            "chip_border": "#b0bec5",
            "chip_color": "#37474f",
            "product_bg": "#ffffff",
            "product_bg_sel": "#e0f2f1",
            "product_border": "#cfd8dc",
            "product_border_sel": "#00695c",
        }


# ===========================================================================
# Data (cached)
# ===========================================================================

@st.cache_data(show_spinner=False)
def get_products():
    return load_products_balanced(n=50)


_CAT_COLORS: dict = {
    "Toys":        ("#fff8e1", "#f57f17"),
    "Electronics": ("#e3f2fd", "#1565c0"),
    "Clothing":    ("#fce4ec", "#c2185b"),
    "Books":       ("#f3e5f5", "#7b1fa2"),
    "Home":        ("#e8f5e9", "#2e7d32"),
    "Kitchen":     ("#fff3e0", "#e65100"),
    "Sports":      ("#e8eaf6", "#283593"),
    "Health":      ("#e0f7fa", "#00695c"),
    "Beauty":      ("#fce4ec", "#880e4f"),
    "Baby":        ("#e1f5fe", "#0277bd"),
    "Arts":        ("#f9fbe7", "#558b2f"),
    "Office":      ("#eceff1", "#37474f"),
    "Pet":         ("#fff3e0", "#bf360c"),
    "Garden":      ("#f1f8e9", "#33691e"),
    "Grocery":     ("#f1f8e9", "#33691e"),
    "Hobbies":     ("#fffde7", "#f9a825"),
    "Music":       ("#ede7f6", "#4527a0"),
    "Auto":        ("#e8eaf6", "#1a237e"),
}


def _cat_colors(category: str):
    """Return (bg_hex, fg_hex) for a top-level category string."""
    for k, v in _CAT_COLORS.items():
        if k.lower() in category.lower():
            return v
    return ("#f5f5f5", "#424242")


@st.cache_data(show_spinner=False)
def get_personas():
    return load_personas()


# ===========================================================================
# Navigation helpers
# ===========================================================================

def go_to(step):
    st.session_state.step = step
    st.rerun()


def next_step():
    go_to(st.session_state.step + 1)


_LOG_PATH = _SHOWCASE_DIR / "data_collect" / "showcase_log.jsonl"
_OUTPUT_DIR = _SHOWCASE_DIR / "output"


_GSHEET_HEADERS = [
    "timestamp", "session_id",
    # Customer persona
    "persona_name", "persona_id", "persona_job_sector", "persona_location",
    "persona_income_range", "persona_description",
    # Agent persona
    "agent_persona", "agent_persona_label",
    # Scenario / task
    "order_id", "purchase_date", "delivery_date",
    "items", "return_reasons",
    "scenario_detail", "scenario_return_details", "policy_ambiguities",
    "complexity_level", "narrative",
    # Conversation
    "conversation",
    # Resolution
    "resolution_type", "resolution_description", "resolution_conditions",
    "customer_next_steps",
    # Stats
    "turns", "tool_calls", "hints_used",
]


def _append_to_gsheet(record: dict):
    """Append one summary row to the Google Sheet configured in Streamlit secrets."""
    try:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]),
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        gc = gspread.authorize(creds)
        spreadsheet_id = st.secrets["google_sheets"]["spreadsheet_id"]
        sheet = gc.open_by_key(spreadsheet_id).sheet1

        task = record.get("task", {})
        user = record.get("user", {})
        stats = record.get("stats", {})
        res = record.get("resolution") or {}
        agent = record.get("agent", {})
        scenario_raw = record.get("scenario_raw", {})

        items_str = ", ".join(
            i.get("product_name", "") if isinstance(i, dict) else str(i)
            for i in task.get("items", [])
        )
        conversation_str = "\n".join(
            f"[{m['role'].upper()}] {m['text']}"
            for m in record.get("conversation", [])
        )
        ambiguities = scenario_raw.get("policy_ambiguities") or []
        if isinstance(ambiguities, list):
            ambiguities_str = "; ".join(
                a.get("ambiguity", a) if isinstance(a, dict) else str(a)
                for a in ambiguities
            )
        else:
            ambiguities_str = str(ambiguities)

        row = [
            record.get("timestamp", ""),
            record.get("session_id", ""),
            user.get("name", ""),
            user.get("persona_id", ""),
            user.get("job_sector", ""),
            user.get("location", ""),
            user.get("income_range", ""),
            user.get("persona_description", ""),
            agent.get("key", ""),
            agent.get("label", ""),
            task.get("order_id", ""),
            task.get("purchase_date", ""),
            task.get("delivery_date", ""),
            items_str,
            ", ".join(task.get("return_reasons", [])),
            scenario_raw.get("detail", ""),
            scenario_raw.get("return_details", ""),
            ambiguities_str,
            scenario_raw.get("complexity_level", ""),
            record.get("narrative", ""),
            conversation_str,
            res.get("resolution_type", "") if isinstance(res, dict) else "",
            res.get("resolution_description", "") if isinstance(res, dict) else "",
            "; ".join(res.get("conditions", [])) if isinstance(res, dict) else "",
            res.get("customer_next_steps", "") if isinstance(res, dict) else "",
            stats.get("turn_count", ""),
            stats.get("tool_calls_count", ""),
            stats.get("hints_used", ""),
        ]

        if not sheet.row_values(1):
            sheet.append_row(_GSHEET_HEADERS)
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Could not save to Google Sheet: {e}")


def _save_session_data():
    """Save user, task, conversation and resolution to a JSON file and Google Sheet."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    persona = st.session_state.get("persona") or {}
    first_name = (persona.get("Name") or "unknown").split()[0].lower()
    session_id = st.session_state.get("api_session_id") or st.session_state.get("session_id", "unknown")
    filename = f"{first_name}_{session_id}.json"
    scenario = st.session_state.get("scenario") or {}
    task = scenario.get("task", {})
    agent_key = st.session_state.get("agent_persona_choice", "FAIR")
    agent_label = AGENT_INFO.get(agent_key, {}).get("label", agent_key)
    record = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "user": {
            "name": persona.get("Name", ""),
            "location": persona.get("Location", ""),
            "job_sector": persona.get("Job_sector", ""),
            "income_range": persona.get("Income_range", ""),
            "persona_id": persona.get("Persona_id", ""),
            "persona_description": persona.get("Person description", ""),
        },
        "agent": {
            "key": agent_key,
            "label": agent_label,
        },
        "task": {
            "order_id": task.get("order_id", ""),
            "items": st.session_state.get("selected_items", task.get("items", [])),
            "return_reasons": st.session_state.get("return_reasons", []),
            "purchase_date": task.get("purchase_date", ""),
            "delivery_date": task.get("delivery_date", ""),
        },
        "scenario_raw": {
            "detail": task.get("detail", ""),
            "return_details": task.get("return_details", ""),
            "policy_ambiguities": task.get("policy_ambiguities", []),
            "complexity_level": task.get("complexity_level", ""),
        },
        "narrative": st.session_state.get("narrative", ""),
        "conversation": [
            {
                "role": m["role"],
                "text": m["text"],
                "tool_calls": m.get("tools", []),
            }
            for m in st.session_state.get("messages", [])
        ],
        "resolution": st.session_state.get("resolution"),
        "stats": {
            "turn_count": st.session_state.get("turn_count", 0),
            "tool_calls_count": st.session_state.get("tool_calls_count", 0),
            "hints_used": sum(st.session_state.get("turn_hint_flags", [])),
        },
    }
    out_path = _OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    _append_to_gsheet(record)


def _log_event(event_type: str, data: dict):
    """Append one JSON line to the showcase interaction log."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "session_id": st.session_state.get("session_id", "unknown"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": event_type,
        **data,
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ===========================================================================
# Sidebar — progress + language toggle + (Step 5) hints panel
# ===========================================================================

AGENT_ICONS = {
    "DIRECT": "⚡",
    "FAIR": "⚖️",
    "AGREEABLE": "🤝",
    "HELPFUL": "🌟",
    "VERY_HELPFUL": "💎",
}

AGENT_INFO = {
    "DIRECT": {
        "label": "Direct",
        "one_line": "Hard · Strictly policy-based, concise, no exceptions.",
        "one_line_kid": "Hard · Follows the rules exactly — no special treatment!",
        "diff_color": "#ef5350",
    },
    "FAIR": {
        "label": "Fair",
        "one_line": "Medium · Balanced and procedural, every case by the book.",
        "one_line_kid": "Medium · Fair and careful — treats everyone the same way.",
        "diff_color": "#ff9800",
    },
    "AGREEABLE": {
        "label": "Agreeable",
        "one_line": "Easy · Warm and empathetic, looks for compromise.",
        "one_line_kid": "Easy · Super friendly and kind, tries to help you out.",
        "diff_color": "#66bb6a",
    },
    "HELPFUL": {
        "label": "Helpful",
        "one_line": "Easy · Proactive and thorough, explains all options.",
        "one_line_kid": "Easy · Really helpful — explains everything and finds solutions.",
        "diff_color": "#42a5f5",
    },
    "VERY_HELPFUL": {
        "label": "Very Helpful",
        "one_line": "Very Easy · Customer-first, flexible, enthusiastic.",
        "one_line_kid": "Very Easy · Goes above and beyond to make you happy!",
        "diff_color": "#ce93d8",
    },
}

with st.sidebar:
    # Language toggle
    col_title, col_lang = st.columns([3, 1])
    with col_title:
        st.markdown(f'<div class="px-heading">{t("title")}</div>', unsafe_allow_html=True)
    with col_lang:
        if st.button(t("lang_toggle"), key="lang_btn", help="Switch language"):
            st.session_state.lang = "de" if st.session_state.lang == "en" else "en"
            st.rerun()

    # Kid mode toggle
    kid_label = t("kid_mode_on") if st.session_state.kid_mode else t("kid_mode_off")
    if st.button(kid_label, key="kid_mode_btn", use_container_width=True):
        st.session_state.kid_mode = not st.session_state.kid_mode
        st.rerun()

    st.markdown("---")

    # Step progress
    current_step = st.session_state.step
    for i, label in enumerate(t("step_labels"), start=1):
        icon = "✅" if i < current_step else ("▶" if i == current_step else "·")
        st.markdown(f"{'**' if i == current_step else ''}{icon} {label}{'**' if i == current_step else ''}")

    st.markdown("---")

    # Step 5: character info panel
    if st.session_state.step == 5 and not st.session_state.finished:
        persona = st.session_state.persona
        turn = st.session_state.turn_count
        st.markdown(f'<div class="px-heading" style="font-size:0.7rem;">💡 {t("helper_title")}</div>', unsafe_allow_html=True)
        if persona:
            st.markdown(f"**{t('your_character')}:** {persona.get('Name','?')}")

        # Turn progress bar
        filled = min(turn, 10)
        bar = "■" * filled + "□" * (10 - filled)
        st.markdown(f'<div class="px-bar">{bar}</div>', unsafe_allow_html=True)
        st.caption(f"{t('turn')}: {turn} / 10")

        st.markdown("---")

    if st.button(f"🔄 {t('restart')}", use_container_width=True, key="restart_btn"):
        sid = st.session_state.get("api_session_id")
        if sid:
            try:
                httpx.delete(f"{_API_BASE}/api/session/{sid}", timeout=5)
            except Exception:
                pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ===========================================================================
# Pixel trait bar helper
# ===========================================================================

def trait_bar(score, label, max_score=5):
    filled = int(score)
    bar = "■" * filled + "□" * (max_score - filled)
    return f'<div class="px-card-trait">{bar} <span style="color:#90a4ae;font-size:0.68rem;">{label}</span></div>'



# ===========================================================================
# EXIT BUTTON — top-right on every page
# ===========================================================================

_, _exit_col = st.columns([6, 1])
with _exit_col:
    if st.button("✕ Exit", key="exit_btn_top", help="Abort and return to start", use_container_width=True):
        sid = st.session_state.get("api_session_id")
        if sid:
            try:
                httpx.delete(f"{_API_BASE}/api/session/{sid}", timeout=5)
            except Exception:
                pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ===========================================================================
# STEP 1 — Persona Design
# ===========================================================================

if st.session_state.step == 1:
    st.markdown(f'<div class="px-heading">{t("step1_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="px-sub">{t("step1_sub")}</div>', unsafe_allow_html=True)

    personas = get_personas()
    preset_personas = personas[:6]

    st.markdown(f"### {t('preset_avatars')}")

    th = get_theme()
    cols_per_row = 3
    for row_start in range(0, len(preset_personas), cols_per_row):
        row_ps = preset_personas[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, row_ps):
            with col:
                is_sel = (
                    st.session_state.persona is not None
                    and st.session_state.persona.get("Persona_id") == p.get("Persona_id")
                    and not st.session_state.custom_persona_mode
                )
                border = f"4px solid {th['card_border_sel']}" if is_sel else f"4px solid {th['card_border']}"
                bg = th["card_bg_sel"] if is_sel else th["card_bg"]

                personality = p.get("Personality Style", {})
                agr = personality.get("Agreeableness", 3)
                ner = personality.get("Neuroticism", 3)
                full_desc = p.get("Person description", "")

                st.markdown(
                    f'<div class="px-card-wrap" style="border:{border}; background:{bg}; border-radius:4px; '
                    f'padding:10px 8px; text-align:center; min-height:340px;">'
                    + (f'<div class="px-card-tooltip">{full_desc}</div>' if full_desc else "")
                    + persona_img_html(p, height=180)
                    + f'<div class="px-card-name">{p["Name"]}</div>'
                    f'<div class="px-card-sub">{p.get("Age-range","?")} · {p.get("Location","?")}</div>'
                    f'<div class="px-card-sub">{p.get("Job Sector","")}</div>'
                    + trait_bar(agr, trait_name("Agreeableness"))
                    + trait_bar(ner, trait_name("Neuroticism"))
                    + "</div>",
                    unsafe_allow_html=True,
                )
                label = f"✅ {t('selected')}" if is_sel else t("select")
                if st.button(label, key=f"sel_p_{p.get('Persona_id','')}", use_container_width=True):
                    st.session_state.persona = p
                    st.session_state.custom_persona_mode = False
                    st.rerun()

    st.markdown("---")
    st.markdown(f"### 🎨 {t('create_own')}")

    with st.expander(t("custom_expand"), expanded=st.session_state.custom_persona_mode):
        c1, c2 = st.columns(2)
        with c1:
            cname = st.text_input("Name", value="Alex Kim", key="custom_name")
            cage = st.selectbox("Age Range", ["18-25", "25-30", "30-35", "35-40", "40-50", "50-60", "60+"], key="custom_age")
            clocation = st.text_input("Location", value="United States", key="custom_loc")
        with c2:
            cgender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"], key="custom_gender")
            cjob = st.text_input("Job Sector", value="Technology", key="custom_job")
            cincome = st.text_input("Income Range", value="50,000-75,000 USD", key="custom_income")

        clang = st.selectbox(
            "Communication Style",
            ["Consultative", "Formal", "Casual", "Assertive", "Empathetic"],
            key="custom_lang",
        )

        st.markdown(f"**{t('personality')} Traits**")
        _trait_help = {
            "Openness":          "How curious and open to new ideas is this person? High = loves novelty; Low = prefers routine.",
            "Conscientiousness": "How organised and reliable is this person? High = very disciplined; Low = more spontaneous.",
            "Extraversion":      "How energised by social interaction? High = outgoing and talkative; Low = quiet and reserved.",
            "Agreeableness":     "How cooperative and easy-going? High = warm and accommodating; Low = more direct and skeptical.",
            "Neuroticism":       "How emotionally reactive under stress? High = anxious or worried; Low = calm and steady.",
        }

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            copen = st.slider(trait_name("Openness"), 1, 5, 3, key="custom_open",
                              help=_trait_help["Openness"])
            ccons = st.slider(trait_name("Conscientiousness"), 1, 5, 3, key="custom_cons",
                              help=_trait_help["Conscientiousness"])
        with pc2:
            cextr = st.slider(trait_name("Extraversion"), 1, 5, 3, key="custom_extr",
                              help=_trait_help["Extraversion"])
            cagr = st.slider(trait_name("Agreeableness"), 1, 5, 3, key="custom_agr",
                             help=_trait_help["Agreeableness"])
        with pc3:
            cneur = st.slider(trait_name("Neuroticism"), 1, 5, 2, key="custom_neur",
                              help=_trait_help["Neuroticism"])

        if st.button(f"✅ Use This Character", key="use_custom_persona"):
            st.session_state.persona = {
                "Name": cname,
                "Age-range": cage,
                "Location": clocation,
                "Gender": cgender,
                "Job Sector": cjob,
                "Income range": cincome,
                "Language Style": clang,
                "Personality Style": {
                    "Openness": copen,
                    "Conscientiousness": ccons,
                    "Extraversion": cextr,
                    "Agreeableness": cagr,
                    "Neuroticism": cneur,
                },
                "Communication style": clang,
                "Persona_id": f"custom_{cname.replace(' ', '_').lower()}",
            }
            st.session_state.custom_persona_mode = True
            st.rerun()

    st.markdown("---")
    if st.session_state.persona:
        p = st.session_state.persona
        # Show the assigned character image for custom personas
        if st.session_state.get("custom_persona_mode", False):
            st.markdown(persona_img_html(p, height=100), unsafe_allow_html=True)
        st.success(f"{t('avatar_selected')}: **{p['Name']}** ({p.get('Job Sector','')}, {p.get('Location','')})")
        if st.button(f"{t('next_items')} →", type="primary", use_container_width=True):
            _log_event("persona_selected", {
                "persona_id": p.get("Persona_id", ""),
                "persona_name": p.get("Name", ""),
                "persona_age_range": p.get("Age-range", ""),
                "persona_location": p.get("Location", ""),
                "persona_gender": p.get("Gender", ""),
                "persona_job_sector": p.get("Job Sector", ""),
                "persona_is_custom": st.session_state.get("custom_persona_mode", False),
                "personality": p.get("Personality Style", {}),
                "kid_mode": st.session_state.get("kid_mode", False),
                "language": st.session_state.get("lang", "en"),
            })
            next_step()
    else:
        st.info(t("please_select_persona"))


# ===========================================================================
# STEP 2 — Product Selection
# ===========================================================================

elif st.session_state.step == 2:
    st.markdown(f'<div class="px-heading">{t("step2_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="px-sub">{t("step2_sub")}</div>', unsafe_allow_html=True)

    products = get_products()
    categories = [t("all_categories")] + get_product_categories(products)

    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍", key="product_search", placeholder=t("search_placeholder"), label_visibility="collapsed")
    with col_cat:
        selected_cat = st.selectbox(t("category"), categories, key="product_category", label_visibility="collapsed")

    filtered = products
    if search_query:
        q = search_query.lower()
        filtered = [p for p in filtered if q in p.get("product_name", "").lower() or q in p.get("description", "").lower()]
    if selected_cat != t("all_categories"):
        filtered = [p for p in filtered if p.get("category", "").startswith(selected_cat)]

    display_products = filtered[:50]
    n_sel = len(st.session_state.selected_items)

    # Items counter pixel bar
    bar = "■" * n_sel + "□" * (3 - n_sel)
    st.markdown(
        f'<div class="px-bar" style="margin-bottom:8px;">{bar} '
        f'<span style="color:#90a4ae;font-size:0.78rem;font-family:sans-serif;">'
        f'{n_sel} / 3 {t("items_counter")}</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{t('showing')} {len(display_products)} {t('of')} {len(filtered)} {t('products')}")

    RETURN_REASONS_LIST = get_reasons()
    current_selected = {item["product_name"]: item for item in st.session_state.selected_items}

    _ICON_MAP = {
        "Toys": "🧸", "Electronics": "📱", "Clothing": "👕", "Books": "📚",
        "Home": "🏠", "Kitchen": "🍳", "Sports": "⚽", "Garden": "🌿",
        "Health": "💊", "Beauty": "💄", "Auto": "🚗", "Tools": "🔧",
        "Food": "🍎", "Grocery": "🛒", "Music": "🎵", "Office": "📎",
        "Pet": "🐾", "Baby": "🍼", "Hobbies": "🎨", "Arts": "✂️",
    }

    def _prod_icon(cat):
        for k, v in _ICON_MAP.items():
            if k.lower() in cat.lower():
                return v
        return "📦"

    th = get_theme()
    grid_cols = st.columns(3)
    for idx, product in enumerate(display_products):
        pname = product["product_name"]
        is_checked = pname in current_selected
        price = product.get("selling_price", "N/A")
        cat_full = product.get("category", "")
        cat_short = cat_full.split("|")[0].strip()
        icon = _prod_icon(cat_full)
        cat_bg, cat_fg = _cat_colors(cat_full)
        desc = product.get("description", "")[:110]
        is_amz = product.get("is_amazon_seller") == "Y"

        card_border = f"3px solid {th['product_border_sel']}" if is_checked else f"2px solid {th['product_border']}"
        card_bg = th["product_bg_sel"] if is_checked else th["product_bg"]
        sel_ring = f"box-shadow:0 0 0 3px {th['product_border_sel']};" if is_checked else "box-shadow:0 2px 8px rgba(0,0,0,0.07);"

        seller_html = (
            f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
            f'font-size:0.62rem;font-weight:700;background:#e65100;color:#fff;">Amazon</span>'
            if is_amz else
            f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
            f'font-size:0.62rem;font-weight:700;background:#546e7a;color:#fff;">3rd Party</span>'
        )

        col = grid_cols[idx % 3]
        with col:
            st.markdown(
                f'<div style="border-radius:10px;overflow:hidden;border:{card_border};'
                f'background:{card_bg};{sel_ring}margin-bottom:2px;">'
                # ── category header band ──
                f'<div style="background:{cat_bg};padding:18px 8px 12px;text-align:center;">'
                f'<div style="font-size:2.6rem;line-height:1;">{icon}</div>'
                f'<div style="margin-top:4px;font-size:0.6rem;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:{cat_fg};">{cat_short}</div>'
                f'</div>'
                # ── product body ──
                f'<div style="padding:10px 12px 12px;text-align:left;">'
                f'<div style="font-weight:700;font-size:0.84rem;color:{th["card_text"]};'
                f'line-height:1.3;margin-bottom:6px;min-height:2.6em;">'
                f'{pname[:70]}{"…" if len(pname)>70 else ""}</div>'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">'
                f'<span style="font-size:1rem;font-weight:700;color:#00695c;">{price}</span>'
                f'{seller_html}</div>'
                f'<div style="font-size:0.71rem;color:{th["card_sub"]};line-height:1.4;">'
                f'{desc}{"…" if len(product.get("description",""))>110 else ""}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            checked = st.checkbox(
                f"{'✅ ' if is_checked else ''}{t('select')}",
                value=is_checked,
                key=f"chk_{idx}_{pname[:30]}",
                disabled=(not is_checked and n_sel >= 3),
            )
            if checked and not is_checked:
                st.session_state.selected_items.append(product)
                st.session_state.return_reasons.append(RETURN_REASONS_LIST[-1])  # default: "Other"
                st.rerun()
            elif not checked and is_checked:
                item_idx = next(
                    (i for i, x in enumerate(st.session_state.selected_items) if x["product_name"] == pname), None
                )
                if item_idx is not None:
                    st.session_state.selected_items.pop(item_idx)
                    st.session_state.return_reasons.pop(item_idx)
                    st.rerun()

            if is_checked:
                item_idx = next(
                    (i for i, x in enumerate(st.session_state.selected_items) if x["product_name"] == pname), 0
                )
                cur_reason = st.session_state.return_reasons[item_idx] if item_idx < len(st.session_state.return_reasons) else RETURN_REASONS_LIST[-1]
                new_reason = st.selectbox(
                    t("return_reason"),
                    RETURN_REASONS_LIST,
                    index=RETURN_REASONS_LIST.index(cur_reason) if cur_reason in RETURN_REASONS_LIST else 0,
                    key=f"reason_{idx}_{pname[:30]}",
                )
                if new_reason != cur_reason and item_idx < len(st.session_state.return_reasons):
                    st.session_state.return_reasons[item_idx] = new_reason

    if st.session_state.selected_items:
        st.markdown("---")
        st.markdown(f"#### ✅ {t('selected_items')}")
        for item, reason in zip(st.session_state.selected_items, st.session_state.return_reasons):
            st.markdown(f"- **{item['product_name']}** — {reason}")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            go_to(1)
    with col_next:
        if st.session_state.selected_items:
            if st.button(f"{t('next_agent')} →", type="primary", use_container_width=True):
                _log_event("items_selected", {
                    "persona_id": st.session_state.persona.get("Persona_id", "") if st.session_state.persona else "",
                    "items": [
                        {
                            "product_name": item.get("product_name"),
                            "category": item.get("category"),
                            "selling_price": item.get("selling_price"),
                            "is_amazon_seller": item.get("is_amazon_seller"),
                        }
                        for item in st.session_state.selected_items
                    ],
                    "return_reasons": st.session_state.return_reasons,
                })
                next_step()
        else:
            st.info(t("select_at_least_one"))


# ===========================================================================
# STEP 3 — Agent & Model
# ===========================================================================

elif st.session_state.step == 3:
    st.markdown(f'<div class="px-heading">{t("step3_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="px-sub">{t("step3_sub")}</div>', unsafe_allow_html=True)

    st.markdown(f"### {t('agent_personality')}")

    th = get_theme()
    agent_keys = list(AGENT_INFO.keys())
    cols = st.columns(len(agent_keys))
    for col, persona_key in zip(cols, agent_keys):
        info = AGENT_INFO[persona_key]
        is_sel = st.session_state.agent_persona_choice == persona_key
        border = f"4px solid {th['card_border_sel']}" if is_sel else f"4px solid {th['card_border']}"
        bg = th["card_bg_sel"] if is_sel else th["card_bg"]

        with col:
            st.markdown(
                f'<div style="border:{border}; background:{bg}; border-radius:4px; '
                f'padding:10px 8px; text-align:center; min-height:260px;">'
                + agent_img_html(persona_key, height=150)
                + f'<div class="px-card-name">{info["label"]}</div>'
                f'<div class="px-card-sub" style="font-size:0.72rem;color:{info["diff_color"]};">'
                f'{info["one_line_kid"] if st.session_state.get("kid_mode") else info["one_line"]}</div>'
                + "</div>",
                unsafe_allow_html=True,
            )
            label = f"✅ {t('selected')}" if is_sel else t("select")
            if st.button(label, key=f"agent_{persona_key}", use_container_width=True):
                st.session_state.agent_persona_choice = persona_key
                st.rerun()

    st.markdown("---")
    st.markdown(f"### {t('model_label')}")

    MODEL_OPTIONS = [
        "gpt-4.1-2025-04-14",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ]
    MODEL_SUMMARIES = {
        "gpt-4.1-2025-04-14": "GPT-4.1 (default) — OpenAI's latest flagship model. Very capable, follows instructions precisely, and handles complex multi-turn conversations well.",
        "gpt-4o": "GPT-4o — Fast and capable multimodal model from OpenAI. Great balance of speed and reasoning for most conversations.",
        "gpt-4o-mini": "GPT-4o Mini — Lightweight and quick. Lower cost; good for simple cases but may miss policy nuances.",
        "gpt-4-turbo": "GPT-4 Turbo — Previous-generation flagship. Strong reasoning, slightly slower than GPT-4o.",
    }
    col_m, col_custom = st.columns([2, 2])
    with col_m:
        model_sel = st.selectbox(
            t("model_label"),
            MODEL_OPTIONS + ["Custom..."],
            index=MODEL_OPTIONS.index(st.session_state.model_choice)
            if st.session_state.model_choice in MODEL_OPTIONS
            else len(MODEL_OPTIONS),
            key="model_select",
            label_visibility="collapsed",
        )
    with col_custom:
        if model_sel == "Custom...":
            st.session_state.model_choice = st.text_input(
                "Custom model name", value=st.session_state.model_choice, key="custom_model_input"
            )
        else:
            st.session_state.model_choice = model_sel

    summary = MODEL_SUMMARIES.get(st.session_state.model_choice)
    if summary:
        st.caption(f"ℹ️ {summary}")

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            go_to(2)
    with col_next:
        if st.button(f"{t('next_mission')} →", type="primary", use_container_width=True):
            _log_event("agent_selected", {
                "persona_id": st.session_state.persona.get("Persona_id", "") if st.session_state.persona else "",
                "agent_persona": st.session_state.agent_persona_choice,
                "model": st.session_state.model_choice,
            })
            en_reasons = [reason_to_en(r) for r in st.session_state.return_reasons]
            # Pre-generate one order ID per item so multi-item scenarios can have separate IDs
            import random as _random
            scenario_order_ids = [f"AMZ-{_random.randint(1000000, 9999999)}" for _ in st.session_state.selected_items]
            # Generate LLM task (complex scenario with ambiguities)
            with st.spinner("🧩 Generating your mission..."):
                task_provider = make_provider(model=st.session_state.model_choice, temperature=0.9, max_tokens=900)
                task_detail = generate_task_detail(
                    selected_items=st.session_state.selected_items,
                    persona=st.session_state.persona,
                    provider=task_provider,
                    return_reasons=en_reasons,
                    order_ids=scenario_order_ids,
                )
            scenario = build_scenario(
                persona=st.session_state.persona,
                selected_items=st.session_state.selected_items,
                return_reasons=en_reasons,
                task_detail=task_detail,
                order_ids=scenario_order_ids,
            )
            st.session_state.scenario = scenario
            # Clear narrative so Step 4 regenerates
            st.session_state.narrative = ""
            st.session_state.starters = []
            next_step()


# ===========================================================================
# STEP 4 — Narrative + Conversation Starters
# ===========================================================================

elif st.session_state.step == 4:
    st.markdown(f'<div class="px-heading">{t("step4_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="px-sub">{t("step4_sub")}</div>', unsafe_allow_html=True)

    scenario = st.session_state.scenario

    # Generate narrative + starters once on arrival
    if not st.session_state.narrative:
        with st.spinner(t("generating")):
            provider = make_provider(model=st.session_state.model_choice, temperature=0.8, max_tokens=400)
            kid = st.session_state.get("kid_mode", False)
            st.session_state.narrative = generate_narrative(scenario, provider, kid_mode=kid)
            st.session_state.starters = generate_starters(scenario, provider, kid_mode=kid)

    # Persona card at top of mission page
    if st.session_state.persona:
        p = st.session_state.persona
        persona_desc = p.get("Person description", "")
        col_pimg, col_pinfo = st.columns([1, 3])
        with col_pimg:
            st.markdown(persona_img_html(p, height=110), unsafe_allow_html=True)
        with col_pinfo:
            st.markdown(f"**{p['Name']}** · {p.get('Age-range', '')} · {p.get('Location', '')}")
            st.markdown(f"*{p.get('Job Sector', '')}*")
            if persona_desc:
                st.markdown(persona_desc)
        st.markdown("---")

    st.markdown(
        f'<div class="px-narrative">'
        f'<div class="px-narrative-title">📜 YOUR MISSION</div>'
        f'<p><strong>You have submitted your return request form. You are now contacting the Customer Service Agent to discuss and resolve your return.</strong></p>'
        f'{st.session_state.narrative}'
        f'</div>',
        unsafe_allow_html=True,
    )

    task = scenario.get("task", {})
    customer_agent_info = task.get("customer_agent_info", "")
    if customer_agent_info:
        st.info(f"**Your stated issue:** {customer_agent_info}")

    with st.expander("📋 Full scenario details"):
        st.markdown(task.get("detail", "*(not available)*"))

    basic_info_s4 = task.get("basic_info", {})
    return_details_s4 = task.get("return_details", "")
    with st.expander("📋 Your case facts"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            order_ids_s4 = basic_info_s4.get("order_ids") or [basic_info_s4.get("order_id", task.get("order_id", "—"))]
            for oid in order_ids_s4:
                st.markdown(f"**Order ID:** {oid}")
            st.markdown(f"**Purchase date:** {basic_info_s4.get('order_date', task.get('purchase_date', '—'))}")
            st.markdown(f"**Delivery date:** {basic_info_s4.get('delivery_date', task.get('delivery_date', '—'))}")
        with col_f2:
            products_s4 = basic_info_s4.get("products", [])
            if products_s4:
                for prod in products_s4:
                    st.markdown(f"**{prod.get('product_name', '—')}**")
            else:
                for item in task.get("items", []):
                    st.markdown(f"**{item.get('product_name', '—')}**")
        if return_details_s4:
            st.markdown("---")
            st.markdown(f"**Your case:** {return_details_s4}")

    st.markdown(f"**{t('how_to_start')}**")

    starter_options = st.session_state.starters + [f"✏️ {t('write_own')}"]
    choice = st.radio("", starter_options, key="starter_choice", label_visibility="collapsed")

    if choice == starter_options[-1]:  # "Write my own"
        custom_msg = st.text_area(
            "",
            value=st.session_state.first_message,
            height=120,
            placeholder=t("type_reply"),
            key="custom_first_msg",
            label_visibility="collapsed",
        )
        st.session_state.first_message = custom_msg
    else:
        st.session_state.first_message = choice

    st.markdown("---")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", use_container_width=True):
            go_to(3)
    with col_next:
        msg = st.session_state.first_message.strip() if st.session_state.first_message else ""
        if msg:
            if st.button(f"{t('begin_chat')} →", type="primary", use_container_width=True):
                scenario["first_customer_message"] = msg
                st.session_state.scenario = scenario

                with st.spinner(t("agent_thinking")):
                    try:
                        resp = httpx.post(
                            f"{_API_BASE}/api/session",
                            json={
                                "scenario": scenario,
                                "agent_persona": st.session_state.agent_persona_choice,
                                "model": st.session_state.model_choice,
                                "first_message": msg,
                            },
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Could not start conversation: {e}")
                        st.stop()

                st.session_state.api_session_id = data["session_id"]
                st.session_state.tool_calls_count = data.get("tool_calls_count", 0)
                st.session_state.messages = [
                    {"role": "customer", "text": msg, "tools": []},
                    {
                        "role": "agent",
                        "text": data["agent_message"],
                        "tools": data.get("tool_calls", []),
                        "reasoning": data.get("reasoning"),
                    },
                ]
                st.session_state.hint_text = data.get("hint")
                _log_event("conversation_started", {
                    "persona_id": st.session_state.persona.get("Persona_id", "") if st.session_state.persona else "",
                    "agent_persona": st.session_state.agent_persona_choice,
                    "model": st.session_state.model_choice,
                    "first_message": msg,
                })
                st.session_state.turn_count = 1
                st.session_state.finished = data.get("finished", False)
                st.session_state.resolution = data.get("resolution")
                st.session_state.accepted_resolution = False
                st.session_state.turn_hint_flags = []
                st.session_state.hint_used_this_turn = False
                if data.get("verification_result"):
                    st.session_state.verification_result = data["verification_result"]
                next_step()
        else:
            st.info(t("write_msg_first"))


# ===========================================================================
# STEP 5 — Conversation
# ===========================================================================

elif st.session_state.step == 5:
    persona = st.session_state.persona
    agent_persona = st.session_state.agent_persona_choice
    info = AGENT_INFO[agent_persona]

    # Header row
    th = get_theme()
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        sprite_key = get_persona_sprite_key(persona)
        sprite = PERSONA_SPRITES.get(sprite_key, PERSONA_SPRITES["default_m"])
        sprite_html = pixel_sprite_html(sprite, px=5, grid_w=10, grid_h=18)
        st.markdown(
            sprite_html
            + f'<div style="text-align:center;font-weight:700;color:{th["card_text"]};">{persona["Name"]}</div>'
            f'<div style="text-align:center;font-size:0.78rem;color:{th["card_sub"]};">'
            f'{persona.get("Job Sector","")}, {persona.get("Location","")}</div>',
            unsafe_allow_html=True,
        )
    with col_h2:
        agent_sprite = AGENT_SPRITES[agent_persona]
        agent_sprite_html = pixel_sprite_html(agent_sprite, px=5, grid_w=10, grid_h=15)
        st.markdown(
            agent_sprite_html
            + f'<div style="text-align:center;font-weight:700;color:{th["card_text"]};">'
            f'{AGENT_ICONS[agent_persona]} {info["label"]} Agent</div>'
            f'<div style="text-align:center;font-size:0.78rem;color:{th["card_sub"]};">'
            f'{st.session_state.model_choice}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Task facts panel — always accessible during chat
    scenario = st.session_state.scenario
    task = scenario.get("task", {})
    basic_info = task.get("basic_info", {})
    return_details = task.get("return_details", "")
    with st.expander("📋 Your case facts", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            order_ids_list = basic_info.get("order_ids") or [basic_info.get("order_id", task.get("order_id", "—"))]
            for oid in order_ids_list:
                st.markdown(f"**Order ID:** {oid}")
            st.markdown(f"**Purchase date:** {basic_info.get('order_date', task.get('purchase_date', task.get('order_date', '—')))}")
            st.markdown(f"**Delivery date:** {basic_info.get('delivery_date', task.get('delivery_date', '—'))}")
        with col_f2:
            products = basic_info.get("products", [])
            if products:
                for prod in products:
                    st.markdown(f"**{prod.get('product_name', '—')}**")
            else:
                for item in task.get("items", []):
                    st.markdown(f"**{item.get('product_name', '—')}**")
        if return_details:
            st.markdown("---")
            st.markdown(f"**Your case:** {return_details}")

    # (Agent turn is triggered by the send button below — not auto-run on rerender)

    # Render conversation
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "customer":
                st.markdown(
                    f'<div class="customer-bubble"><strong>You</strong><br>{msg["text"]}</div>'
                    f'<div class="clearfix"></div>',
                    unsafe_allow_html=True,
                )
            else:
                tool_html = "".join(
                    f'<span class="tool-chip">🔧 {tool["tool_name"]}</span>'
                    for tool in msg.get("tools", [])
                )
                st.markdown(
                    f'<div class="agent-bubble">'
                    f'<strong>{AGENT_ICONS[agent_persona]} {info["label"]} Agent</strong><br>'
                    + (tool_html + "<br>" if tool_html else "")
                    + msg["text"]
                    + "</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                if msg.get("reasoning"):
                    with st.expander("🧠 Agent reasoning (internal)", expanded=False):
                        st.markdown(f"*{msg['reasoning']}*")

    # Resolution reached
    if st.session_state.finished and st.session_state.resolution:
        st.success("✅ The agent has reached a resolution.")
        if st.button(f"{t('view_resolution')} →", type="primary", use_container_width=True):
            go_to(6)

    elif not st.session_state.finished:
        st.divider()
        _prefill = st.session_state.pop("_prefill_msg", "")
        _input_key = f"chat_input_{st.session_state.input_counter}"
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "msg",
                value=_prefill,
                key=_input_key,
                placeholder=t("type_reply"),
                label_visibility="collapsed",
            )
        with col_send:
            send = st.button(f"▶ {t('send')}", type="primary", key="send_btn")

        if send and user_input and user_input.strip():
            msg_text = user_input.strip()
            st.session_state.messages.append({"role": "customer", "text": msg_text, "tools": []})
            st.session_state.turn_hint_flags.append(st.session_state.hint_used_this_turn)
            st.session_state.hint_used_this_turn = False
            st.session_state.turn_count += 1
            st.session_state.hint_text = None

            with st.spinner(f"🤖 {t('agent_thinking')}"):
                try:
                    resp = httpx.post(
                        f"{_API_BASE}/api/session/{st.session_state.api_session_id}/turn",
                        json={"message": msg_text},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.rerun()

            st.session_state.messages.append({
                "role": "agent",
                "text": data["agent_message"],
                "tools": data.get("tool_calls", []),
                "reasoning": data.get("reasoning"),
            })
            st.session_state.tool_calls_count = data.get("tool_calls_count", st.session_state.tool_calls_count)
            st.session_state.hint_text = data.get("hint")
            st.session_state.finished = data.get("finished", False)
            if data.get("resolution"):
                st.session_state.resolution = data["resolution"]
            if data.get("verification_result"):
                st.session_state.verification_result = data["verification_result"]
            st.session_state["input_counter"] += 1
            st.rerun()

        if st.session_state.turn_count >= 9:
            st.warning(t("turn_limit_warning"))

        # Hint button — below chatbox
        col_hint, col_hint_spacer = st.columns([2, 3])
        with col_hint:
            if st.button(f"💡 {t('get_hint')}", key="hint_btn_chat", use_container_width=True):
                with st.spinner(t("hint_generating")):
                    try:
                        resp = httpx.get(
                            f"{_API_BASE}/api/session/{st.session_state.api_session_id}/hint",
                            timeout=35,
                        )
                        if resp.status_code == 200:
                            st.session_state.hint_text = resp.json().get("hint") or "(No hint available)"
                        else:
                            st.session_state.hint_text = "(Hint unavailable)"
                    except Exception as e:
                        st.session_state.hint_text = f"(Hint error: {e})"
                st.rerun()

        if st.session_state.hint_text:
            st.info(f"💡 {st.session_state.hint_text}")
            col_use, col_dismiss = st.columns([2, 2])
            with col_use:
                if st.button(f"↓ {t('use_hint')}", key="use_hint_chat", use_container_width=True):
                    st.session_state.hint_used_this_turn = True
                    st.session_state["_prefill_msg"] = st.session_state.hint_text
                    st.session_state["input_counter"] += 1
                    st.session_state.hint_text = None
                    st.rerun()
            with col_dismiss:
                if st.button("✕ Dismiss", key="dismiss_hint_chat", use_container_width=True):
                    st.session_state.hint_text = None
                    st.rerun()


# ===========================================================================
# STEP 6 — Resolution
# ===========================================================================

elif st.session_state.step == 6:
    st.markdown(f'<div class="px-heading">{t("step6_title")}</div>', unsafe_allow_html=True)

    resolution = st.session_state.resolution
    persona = st.session_state.persona

    if not resolution:
        st.warning(t("no_resolution"))
        if st.button(f"← {t('back_to_chat')}"):
            go_to(5)
        st.stop()

    # Resolution is a plain dict (from API JSON); wrap in a simple accessor
    class _Res:
        def __init__(self, d):
            self._d = d if isinstance(d, dict) else {}
        def __getattr__(self, k):
            return self._d.get(k, "")
    resolution = _Res(resolution)

    res_type = resolution.resolution_type
    display = RESOLUTION_DISPLAY.get(res_type, {
        "label": res_type, "icon": "📋", "color": "gray", "description": "",
    })

    COLOR_BG = {
        "green": "#0d2a0d", "orange": "#2a1a00", "blue": "#0d1a2a",
        "red": "#2a0d0d", "purple": "#1a0d2a", "teal": "#002a2a", "gray": "#1a1a1a",
    }
    COLOR_BORDER = {
        "green": "#4caf50", "orange": "#ff9800", "blue": "#42a5f5",
        "red": "#ef5350", "purple": "#ce93d8", "teal": "#4db6ac", "gray": "#90a4ae",
    }
    SCORE_MAP = {
        "RETURN_REFUND_FULL_BANK": 3,
        "RETURN_REFUND_PARTIAL_BANK": 2,
        "RETURN_REFUND_GIFT_CARD": 2,
        "REPLACEMENT_EXCHANGE": 2,
        "ESCALATE_HUMAN_AGENT": 1,
        "DENY_REFUND": 0,
        "USER_ABORT": 0,
    }
    score = SCORE_MAP.get(res_type, 1)
    score_str = "⭐" * score + "☆" * (3 - score) if score > 0 else "☆☆☆"

    bg = COLOR_BG.get(display["color"], "#1a1a1a")
    border_c = COLOR_BORDER.get(display["color"], "#90a4ae")

    if not st.session_state.accepted_resolution:
        st.markdown(
            f'<div style="border:4px solid {border_c};background:{bg};border-radius:4px;'
            f'padding:24px;text-align:center;margin-bottom:16px;">'
            f'<div style="font-size:2.8rem;">{display["icon"]}</div>'
            f'<div style="font-family:\'Press Start 2P\',monospace;font-size:0.9rem;'
            f'color:#fff;margin:10px 0;">{display["label"]}</div>'
            f'<div style="color:#b0bec5;font-size:0.85rem;">{display["description"]}</div>'
            f'<div style="font-size:1.4rem;margin-top:10px;">{score_str}</div>'
            f'<div style="color:#90a4ae;font-size:0.75rem;">{t("return_score")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"#### 📋 {t('resolution_details')}")
            st.markdown(resolution.resolution_description)
            if resolution.conditions:
                st.markdown(f"**{t('conditions')}:**")
                for cond in resolution.conditions:
                    st.markdown(f"- {cond}")

        with col_r:
            st.markdown(f"#### 📍 {t('next_steps')}")
            if resolution.customer_next_steps:
                st.markdown(resolution.customer_next_steps)

        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(t("turns"), st.session_state.turn_count)
        with col_s2:
            st.metric(t("tool_calls"), st.session_state.tool_calls_count)
        with col_s3:
            st.metric("Difficulty", st.session_state.scenario["task"].get("complexity_level", "—"))

        vr = st.session_state.get("verification_result")
        if vr:
            discrepancies = vr.get("discrepancies", [])
            hints = vr.get("verification_hints", [])
            verified = vr.get("verified", True)
            label = "✅ Claims Consistent" if verified else "⚠️ Inconsistencies Detected"
            with st.expander(f"🔍 Consistency Report — {label}"):
                if discrepancies:
                    st.markdown("**Discrepancies found:**")
                    for d in discrepancies:
                        st.markdown(f"- {d}")
                if hints:
                    st.markdown("**Clarification hints:**")
                    for h in hints:
                        st.markdown(f"- {h}")
                if not discrepancies and not hints:
                    st.markdown("No discrepancies were detected between customer claims and scenario facts.")

        st.markdown("---")
        col_a, col_c = st.columns(2)
        with col_a:
            if st.button(f"✅ {t('accept')}", type="primary", use_container_width=True):
                st.session_state.accepted_resolution = True
                _save_session_data()
                st.rerun()
        with col_c:
            if st.button(f"❌ {t('reject')}", use_container_width=True):
                nudge = "I'm not satisfied with this resolution and would like to discuss further options."
                with st.spinner(f"🤖 {t('agent_thinking')}"):
                    try:
                        resp = httpx.post(
                            f"{_API_BASE}/api/session/{st.session_state.api_session_id}/reject",
                            json={"message": nudge},
                            timeout=120,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Could not continue: {e}")
                        st.stop()

                st.session_state.resolution = None
                st.session_state.finished = False
                st.session_state.messages.append({"role": "customer", "text": nudge, "tools": []})
                st.session_state.messages.append({
                    "role": "agent",
                    "text": data["agent_message"],
                    "tools": data.get("tool_calls", []),
                    "reasoning": data.get("reasoning"),
                })
                st.session_state.tool_calls_count = data.get("tool_calls_count", st.session_state.tool_calls_count)
                st.session_state.hint_text = data.get("hint")
                st.session_state.turn_hint_flags.append(False)
                st.session_state.turn_count += 1
                go_to(5)

    else:
        st.balloons()
        st.success(f"🎉 {t('accepted')}")

        st.markdown(
            f'<div style="border:4px solid {border_c};background:{bg};border-radius:4px;'
            f'padding:20px;margin:12px 0;text-align:center;">'
            f'<div style="font-size:2rem;">{display["icon"]}</div>'
            f'<div style="font-family:\'Press Start 2P\',monospace;font-size:0.85rem;'
            f'color:#fff;margin:8px 0;">{display["label"]}</div>'
            f'<div style="font-size:1.4rem;margin-top:6px;">{score_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"#### 📋 Case Details")
            st.markdown(f"**Character:** {persona['Name']}")
            st.markdown(f"**Order:** `{st.session_state.scenario['task']['order_id']}`")
            st.markdown("**Items Returned:**")
            for item, reason in zip(st.session_state.selected_items, st.session_state.return_reasons):
                st.markdown(f"  - {item['product_name']} — *{reason}*")

        with col_r:
            st.markdown("#### 📊 Stats")
            st.metric(t("turns"), st.session_state.turn_count)
            st.metric(t("tool_calls"), st.session_state.tool_calls_count)
            hints_used = sum(st.session_state.turn_hint_flags)
            st.metric("Hints Used", hints_used)

        st.markdown("---")
        st.markdown(f"#### 📌 {t('resolution_details')}")
        st.markdown(resolution.resolution_description)
        if resolution.customer_next_steps:
            st.markdown(f"**{t('next_steps')}:** {resolution.customer_next_steps}")

        vr = st.session_state.get("verification_result")
        if vr:
            discrepancies = vr.get("discrepancies", [])
            hints = vr.get("verification_hints", [])
            verified = vr.get("verified", True)
            label = "✅ Claims Consistent" if verified else "⚠️ Inconsistencies Detected"
            with st.expander(f"🔍 Consistency Report — {label}"):
                if discrepancies:
                    st.markdown("**Discrepancies found:**")
                    for d in discrepancies:
                        st.markdown(f"- {d}")
                if hints:
                    st.markdown("**Clarification hints:**")
                    for h in hints:
                        st.markdown(f"- {h}")
                if not discrepancies and not hints:
                    st.markdown("No discrepancies were detected between customer claims and scenario facts.")

        with st.expander(f"📜 {t('full_transcript')}"):
            _transcript_info = AGENT_INFO.get(st.session_state.get("agent_persona_choice", "FAIR"), {})
            for msg in st.session_state.messages:
                role_label = f"**{persona['Name']}**" if msg["role"] == "customer" else f"**{_transcript_info.get('label', 'Support')} Agent**"
                st.markdown(f"{role_label}: {msg['text']}")
                if msg.get("tools"):
                    for tool in msg["tools"]:
                        st.markdown(f"&nbsp;&nbsp;🔧 *{tool['tool_name']}*", unsafe_allow_html=True)
                st.markdown("---")

        st.markdown(
            '<a href="https://forms.office.com/e/BTDbBZ6JXZ" target="_blank">'
            '<button style="width:100%;padding:0.6rem;background:#1976d2;color:#fff;'
            'border:none;border-radius:4px;font-family:\'Press Start 2P\',monospace;'
            'font-size:0.7rem;cursor:pointer;margin-bottom:0.5rem;">📝 Take Post-Game Survey</button>'
            '</a>',
            unsafe_allow_html=True,
        )

        if st.button(f"🔄 {t('play_again')}", type="primary", use_container_width=True):
            sid = st.session_state.get("api_session_id")
            if sid:
                try:
                    httpx.delete(f"{_API_BASE}/api/session/{sid}", timeout=5)
                except Exception:
                    pass
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
