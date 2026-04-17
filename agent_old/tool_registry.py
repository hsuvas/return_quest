"""
Tool schema module — exposes agent and customer tools in a standardised format.

Re-exports definitions from ``src/dataset/toolset.py`` and adds helpers for
provider-specific format conversion and prompt injection.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# In standalone return_quest, toolset.py lives in the same agent/ folder
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from toolset import (  # noqa: E402
    ALL_TOOLS,
    CUSTOMER_TOOLS,
    READ_TOOLS,
    WRITE_TOOLS,
    get_tool_by_name,
    get_tools_by_category,
    format_tools_for_prompt,
    format_customer_tools_for_prompt,
)


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_agent_tools() -> List[Dict[str, Any]]:
    """All agent tools (read + write) in OpenAI function-calling format."""
    return ALL_TOOLS


def get_customer_tools() -> List[Dict[str, Any]]:
    """All customer tools in OpenAI function-calling format."""
    return CUSTOMER_TOOLS


def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """Extract function names from a list of tool definitions."""
    return [t["function"]["name"] for t in tools]


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Return the JSON-schema ``parameters`` block for *tool_name*."""
    tool = get_tool_by_name(tool_name)
    if tool:
        return tool["function"].get("parameters", {})
    return None


# ---------------------------------------------------------------------------
# Provider format adapters
# ---------------------------------------------------------------------------

def format_for_provider(
    tools: List[Dict[str, Any]],
    provider: str = "openai",
) -> List[Dict[str, Any]]:
    """Convert tools to a provider-specific representation.

    For OpenAI-compatible providers (including LiteLLM's default) the
    definitions are already in the correct shape.  Extend this function
    if Anthropic / Gemini / etc. need a different schema.
    """
    if provider in ("openai", "azure", "litellm_default"):
        return tools
    # Future: add Anthropic, Gemini, etc.
    return tools


# ---------------------------------------------------------------------------
# Prompt-injection formatter (for providers without native function calling)
# ---------------------------------------------------------------------------

def format_tools_for_prompt_detailed() -> str:
    """Format all agent tool definitions as a text block for prompt injection.

    Mirrors the implementation in
    ``output_collect_no_schema_w_multitype.format_tools_for_prompt_detailed``.
    """
    lines = [
        "## Available Tools",
        "",
        "You have access to the following tools. When you need to use a tool,",
        "include it in the `tool_calls_made` array in your JSON response.",
        "",
    ]

    for tool in ALL_TOOLS:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"**{name}**")
        lines.append(f"  Description: {desc}")
        if params:
            lines.append("  Parameters:")
            for param_name, param_info in params.items():
                req_marker = " (required)" if param_name in required else ""
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                lines.append(
                    f"    - {param_name}{req_marker}: {param_type} — {param_desc}"
                )
        lines.append("")

    lines.extend([
        "### Tool Call Format",
        "",
        "Include tool calls in your response JSON like this:",
        "```json",
        "{",
        '  "tool_calls_made": [',
        "    {",
        '      "tool_name": "get_order_details",',
        '      "tool_call_id": "call_001",',
        '      "arguments": {"order_id": "ORDER-123456"}',
        "    }",
        "  ],",
        "  ... rest of your response ...",
        "}",
        "```",
        "",
        "IMPORTANT: Tool results ARE provided back to you in the conversation",
        "history as tool_result entries. Use them to inform your decisions.",
    ])

    return "\n".join(lines)


def format_customer_tools_for_prompt_detailed() -> str:
    """Format customer tools for prompt injection."""
    lines = [
        "## Available Customer Actions",
        "",
        "You have access to the following tools to perform actions during the conversation:",
        "",
    ]

    for tool in CUSTOMER_TOOLS:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func.get("parameters", {}).get("properties", {})
        required = func.get("parameters", {}).get("required", [])

        lines.append(f"**{name}**")
        lines.append(f"  Description: {desc}")
        if params:
            lines.append("  Parameters:")
            for param_name, param_info in params.items():
                req_marker = " (required)" if param_name in required else ""
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                lines.append(
                    f"    - {param_name}{req_marker}: {param_type} — {param_desc}"
                )
        lines.append("")

    return "\n".join(lines)
