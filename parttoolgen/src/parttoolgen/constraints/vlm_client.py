#!/usr/bin/env python3
"""OpenAI-compatible VLM client for 5.15 part-centric programs.

The client returns raw text only. JSON parsing, schema validation, primitive
lowering, metric geometry, and SimToolReal goal generation remain outside the
model boundary.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class VLMClientError(RuntimeError):
    """Raised when a VLM client cannot produce raw response text."""


def _optional_bool(value: bool | None, env_value: str | None) -> bool | None:
    if value is not None:
        return bool(value)
    if env_value is None:
        return None
    text = str(env_value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return None


@dataclass(frozen=True)
class VLMRawResponse:
    """Raw model response and metadata captured before parser validation."""

    raw_text: str
    metadata: dict[str, Any]


class VLMClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        task_instruction: str,
        context_id: str = "",
        system_prompt: str | None = None,
    ) -> VLMRawResponse:
        """Return raw response text for the caller's bounded parser."""


class OpenAICompatibleVLMClient:
    """Small OpenAI-compatible chat-completions client using stdlib HTTP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        image_path: str | Path | None = None,
        image_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        max_new_tokens: int = 768,
        timeout_s: float = 180.0,
        temperature: float = 0.0,
        max_retries: int = 2,
        enable_thinking: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLM_API_KEY")
        self.model = model or os.environ.get("VLM_MODEL") or "gpt-5.4"
        self.base_url = (base_url or os.environ.get("VLM_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        paths: list[str | Path] = []
        if image_paths:
            paths.extend(image_paths)
        elif image_path:
            paths.append(image_path)
        self.image_paths = [Path(path) for path in paths]
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_retries = max(0, int(max_retries))
        self.enable_thinking = _optional_bool(
            enable_thinking,
            os.environ.get("VLM_ENABLE_THINKING") or os.environ.get("DASHSCOPE_ENABLE_THINKING"),
        )
        if not self.api_key:
            raise VLMClientError("OpenAI-compatible VLM mode requires OPENAI_API_KEY or VLM_API_KEY")
        for path in self.image_paths:
            if not path.exists():
                raise VLMClientError(f"prompt image does not exist: {path}")

    @staticmethod
    def _image_data_url(path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _image_url_content(cls, path: Path) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": cls._image_data_url(path)},
        }

    def _user_content(self, prompt: str) -> Any:
        if not self.image_paths:
            return prompt
        return [
            {"type": "text", "text": prompt},
            *(self._image_url_content(path) for path in self.image_paths),
        ]

    def _responses_input(self, prompt: str) -> Any:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": self._image_data_url(path)}
            for path in self.image_paths
        )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _chat_content_from_body(response_body: str) -> str:
        stripped = response_body.strip()
        if stripped.startswith("data:"):
            chunks: list[str] = []
            for line in stripped.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    continue
                parsed = json.loads(data)
                for choice in parsed.get("choices") or []:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    message = choice.get("message") if isinstance(choice, dict) else None
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        chunks.append(delta["content"])
                    elif isinstance(message, dict) and isinstance(message.get("content"), str):
                        chunks.append(message["content"])
            text = "".join(chunks).strip()
            if text:
                return text
            raise VLMClientError("online VLM chat stream contained no message content")
        parsed = json.loads(response_body)
        raw_text = parsed["choices"][0]["message"]["content"]
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise VLMClientError("online VLM response contained empty chat message content")
        return raw_text

    @staticmethod
    def _responses_content_from_body(response_body: str) -> str:
        parsed = json.loads(response_body)
        output_text = parsed.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        chunks: list[str] = []
        for item in parsed.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        text = "".join(chunks).strip()
        if not text:
            raise VLMClientError("online VLM responses output contained no text")
        return text

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> tuple[str, float]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.time()
        transient_http = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    response_body = response.read().decode("utf-8")
                return response_body, round(time.time() - started, 3)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in transient_http and attempt < self.max_retries:
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
                    continue
                raise VLMClientError(f"online VLM HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, socket.timeout) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
                    continue
                if isinstance(exc, socket.timeout):
                    raise VLMClientError(
                        f"online VLM request timed out after {self.timeout_s:.1f}s"
                    ) from exc
                raise VLMClientError(f"online VLM request failed: {exc}") from exc
        raise VLMClientError("online VLM request failed after retries")

    def generate(
        self,
        *,
        prompt: str,
        task_instruction: str,
        context_id: str = "",
        system_prompt: str | None = None,
    ) -> VLMRawResponse:
        system_message = system_prompt or "Return only the requested JSON object. Do not include prose."
        chat_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": self._user_content(prompt)},
            ],
        }
        if self.enable_thinking is not None:
            chat_payload["enable_thinking"] = self.enable_thinking
        endpoint = "chat/completions"
        try:
            response_body, latency_s = self._post_json(endpoint, chat_payload)
            raw_text = self._chat_content_from_body(response_body)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, VLMClientError) as chat_exc:
            responses_payload = {
                "model": self.model,
                "instructions": system_message,
                "input": self._responses_input(prompt),
                "max_output_tokens": self.max_new_tokens,
            }
            endpoint = "responses"
            response_body, latency_s = self._post_json(endpoint, responses_payload)
            try:
                raw_text = self._responses_content_from_body(response_body)
            except (TypeError, json.JSONDecodeError, VLMClientError) as responses_exc:
                raise VLMClientError(
                    "online VLM response did not contain usable text via chat/completions or responses"
                ) from responses_exc
        return VLMRawResponse(
            raw_text=raw_text,
            metadata={
                "client": "openai_compatible",
                "endpoint": endpoint,
                "model": self.model,
                "base_url": self.base_url,
                "image_paths": [str(path) for path in self.image_paths],
                "latency_s": latency_s,
                "max_new_tokens": self.max_new_tokens,
                "enable_thinking": self.enable_thinking,
                "context_id": context_id,
            },
        )


def build_vlm_client(
    *,
    response_mode: str,
    model: str | None = None,
    base_url: str | None = None,
    image_path: str | Path | None = None,
    image_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    max_new_tokens: int = 768,
    enable_thinking: bool | None = None,
) -> VLMClient:
    """Construct the only 5.15 VLM backend: OpenAI-compatible chat completions."""

    if response_mode != "openai-compatible":
        raise VLMClientError("5.15 supports only response_mode=openai-compatible")
    return OpenAICompatibleVLMClient(
        model=model,
        base_url=base_url,
        image_path=image_path,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
