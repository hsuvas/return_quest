"""
Academic AI API client.

Wraps the Academic AI REST endpoint (https://it-u-api.academic-ai.at)
for chat completions. Used as the primary provider with OpenAI as fallback.
"""

import json
from typing import Any, Dict, List, Optional

import requests


class AcademicAIClient:
    """Client for the Academic AI chat completion API."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = "https://it-u-api.academic-ai.at",
        timeout: tuple = (15, 120),
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Academic AI.

        Parameters match the OpenAI chat completion API so callers can
        pass ``tools``, ``tool_choice``, ``temperature``, etc. directly.

        Returns the raw JSON response dict.

        Raises ``RuntimeError`` on HTTP errors.
        """
        url = f"{self.base_url}/api/v1/llm/chat"
        headers = {
            "X-Client-ID": self.client_id,
            "X-Client-Secret": self.client_secret,
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"model": model, "messages": messages, **kwargs}

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

        if not r.ok:
            raise RuntimeError(
                f"Academic AI HTTP {r.status_code}\nResponse: {r.text}"
            )

        return r.json()
