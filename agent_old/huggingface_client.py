"""
HuggingFace Inference API client.

Uses ``huggingface_hub.InferenceClient`` for chat completions.
Serves as a fallback when LiteLLM does not support the requested model.
"""

import json
import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient


class HuggingFaceClient:
    """Thin wrapper around ``InferenceClient.chat_completion``."""

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 120,
    ):
        token = token or os.environ.get(
            "HUGGINGFACE_API_KEY",
            os.environ.get("HF_TOKEN", ""),
        )
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HUGGINGFACE_API_KEY or HF_TOKEN."
            )
        self.client = InferenceClient(token=token, timeout=timeout)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2500,
        top_p: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Any:
        """Call the HuggingFace Inference API for chat completion.

        Returns a ``ChatCompletionOutput`` object whose structure mirrors
        the OpenAI ``ChatCompletion`` (choices, usage, etc.), so it can be
        parsed with ``LLMResponse.from_openai()``.
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        return self.client.chat_completion(**kwargs)
