"""
LLM runtime / provider adapter — wraps the OpenAI Python client.

Normalises every provider response into an ``LLMResponse`` object so the
rest of the agent system never touches provider-specific types.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm

from academic_ai_client import AcademicAIClient
from huggingface_client import HuggingFaceClient


# ---------------------------------------------------------------------------
# Normalised response
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Provider-agnostic representation of one LLM completion."""

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    # ----- factory -----

    @classmethod
    def from_openai(cls, response: Any) -> "LLMResponse":
        """Build from an OpenAI ``ChatCompletion`` object."""
        choice = response.choices[0]
        content = choice.message.content

        tool_calls = None
        if getattr(choice.message, "tool_calls", None):
            tool_calls = []
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "tool_name": tc.function.name,
                    "tool_call_id": tc.id,
                    "arguments": args,
                })

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        }

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=usage,
        )

    @classmethod
    def from_academic_ai(cls, response: Dict[str, Any]) -> "LLMResponse":
        """Build from an Academic AI JSON response.

        Expected shape::

            {"data": {"content": "...", "usage": {"totalTokens": N, ...}}}

        The response may also include tool_calls when the underlying model
        supports function calling.
        """
        data = response.get("data", {})
        if not isinstance(data, dict):
            data = {}

        content = data.get("content")

        # Parse tool calls if present
        tool_calls = None
        raw_tc = data.get("tool_calls")
        if raw_tc and isinstance(raw_tc, list):
            tool_calls = []
            for tc in raw_tc:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "tool_name": func.get("name", ""),
                    "tool_call_id": tc.get("id", ""),
                    "arguments": args,
                })

        raw_usage = data.get("usage", {})
        usage = {
            "input_tokens": raw_usage.get("promptTokens", raw_usage.get("prompt_tokens", 0)),
            "output_tokens": raw_usage.get("completionTokens", raw_usage.get("completion_tokens", 0)),
        }

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=data.get("finish_reason", "stop") or "stop",
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Provider wrapper
# ---------------------------------------------------------------------------

class LLMProvider:
    """Unified LLM interface backed by the OpenAI Python client."""

    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 2500,
        top_p: float = 1.0,
        fallback_model: Optional[str] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        academic_ai_client: Optional[AcademicAIClient] = None,
        huggingface_client: Optional[HuggingFaceClient] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.academic_ai_client = academic_ai_client
        self.huggingface_client = huggingface_client
        # LiteLLM handles provider routing (openai, huggingface, etc.)

        # Cumulative token counters (thread-safe enough for our purposes)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    # ----- public API -----

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Call the LLM, optionally exposing tools for function calling."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return self._call_with_retry(kwargs)

    def call_text_only(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Call the LLM without tool definitions."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
        }
        return self._call_with_retry(kwargs)

    # ----- internal -----

    def _call_with_retry(self, kwargs: Dict[str, Any]) -> LLMResponse:
        """Try Academic AI first (if configured), then fall back to OpenAI."""
        last_err: Optional[Exception] = None

        # --- Academic AI (primary when configured) ---
        if self.academic_ai_client is not None:
            delay = self.initial_retry_delay
            for attempt in range(self.max_retries + 1):
                try:
                    result = self._call_academic_ai(kwargs)
                    self._record_usage(result.usage)
                    return result
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        jitter = random.uniform(0, delay * 0.5)
                        time.sleep(delay + jitter)
                        delay *= 2
            print(f"[AcademicAI] All {self.max_retries + 1} attempts failed, "
                  f"falling back to OpenAI. Last error: {last_err}")

        # --- LiteLLM (fallback, or sole provider) ---
        delay = self.initial_retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = litellm.completion(**kwargs)
                result = LLMResponse.from_openai(response)
                self._record_usage(result.usage)
                return result
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    jitter = random.uniform(0, delay * 0.5)
                    time.sleep(delay + jitter)
                    delay *= 2

        # --- HuggingFace Inference API (fallback when LiteLLM fails) ---
        if self.huggingface_client is not None:
            print(f"[LiteLLM] All {self.max_retries + 1} attempts failed, "
                  f"falling back to HuggingFace. Last error: {last_err}")
            delay = self.initial_retry_delay
            for attempt in range(self.max_retries + 1):
                try:
                    result = self._call_huggingface(kwargs)
                    self._record_usage(result.usage)
                    return result
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        jitter = random.uniform(0, delay * 0.5)
                        time.sleep(delay + jitter)
                        delay *= 2

        # Fallback model (last resort)
        if self.fallback_model and self.fallback_model != kwargs.get("model"):
            kwargs["model"] = self.fallback_model
            try:
                response = litellm.completion(**kwargs)
                result = LLMResponse.from_openai(response)
                self._record_usage(result.usage)
                return result
            except Exception:
                pass

        raise RuntimeError(
            f"LLM call failed after all attempts. Last error: {last_err}"
        )

    def _call_academic_ai(self, kwargs: Dict[str, Any]) -> LLMResponse:
        """Single Academic AI call. Translates kwargs to the Academic AI API.

        The Academic AI endpoint only accepts ``user`` and ``assistant`` roles
        (no ``system``), and does not support ``max_tokens`` or ``top_p``.
        """
        api_kwargs: Dict[str, Any] = {}
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        # NOTE: max_tokens and top_p are NOT supported by Academic AI API
        if "tools" in kwargs:
            api_kwargs["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            api_kwargs["tool_choice"] = kwargs["tool_choice"]

        # Convert system messages → user messages (API only accepts user/assistant)
        messages = []
        for msg in kwargs["messages"]:
            if msg.get("role") == "system":
                messages.append({"role": "user", "content": msg["content"]})
            else:
                messages.append(msg)

        raw = self.academic_ai_client.create_chat_completion(
            model=kwargs["model"],
            messages=messages,
            **api_kwargs,
        )
        return LLMResponse.from_academic_ai(raw)

    def _call_huggingface(self, kwargs: Dict[str, Any]) -> LLMResponse:
        """Single HuggingFace Inference API call.

        The HF ``ChatCompletionOutput`` mirrors OpenAI's structure, so we
        reuse ``LLMResponse.from_openai`` to parse it.
        """
        hf_kwargs: Dict[str, Any] = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        if "tools" in kwargs:
            hf_kwargs["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            hf_kwargs["tool_choice"] = kwargs["tool_choice"]

        response = self.huggingface_client.chat_completion(**hf_kwargs)
        return LLMResponse.from_openai(response)

    def _record_usage(self, usage: Dict[str, int]) -> None:
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_requests += 1
