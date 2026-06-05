import json
import urllib.request
from typing import Optional

from parttoolgen.constraints.vlm_client import OpenAICompatibleVLMClient


class _FakeHTTPResponse:
    def __init__(self, body: Optional[bytes] = None):
        self._body = body or b'{"choices":[{"message":{"content":"{\\"ok\\": true}"}}]}'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_openai_compatible_vlm_client_attaches_images(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "candidate_sheet.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nunit-test-image")
    captured: dict[str, bytes] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        return _FakeHTTPResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatibleVLMClient(
        api_key="test-key",
        model="vision-model",
        base_url="https://example.invalid/v1",
        image_paths=[image_path],
        max_new_tokens=123,
    )
    response = client.generate(
        prompt="Choose the functional marker.",
        task_instruction="hit the block",
        context_id="unit_context",
        system_prompt="Return JSON only.",
    )

    payload = json.loads(captured["body"].decode("utf-8"))
    user_content = payload["messages"][1]["content"]
    assert payload["model"] == "vision-model"
    assert payload["max_tokens"] == 123
    assert user_content[0] == {"type": "text", "text": "Choose the functional marker."}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert response.raw_text == '{"ok": true}'
    assert response.metadata["image_paths"] == [str(image_path)]
    assert response.metadata["endpoint"] == "chat/completions"


def test_openai_compatible_vlm_client_falls_back_to_responses(monkeypatch) -> None:
    captured: list[bytes] = []

    def fake_urlopen(request, timeout):
        captured.append(request.data)
        if len(captured) == 1:
            return _FakeHTTPResponse(
                b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}\n\n'
                b"data: [DONE]\n\n"
            )
        return _FakeHTTPResponse(
            b'{"output":[{"type":"message","content":[{"type":"output_text","text":"{\\"ok\\": true}"}]}]}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatibleVLMClient(
        api_key="test-key",
        model="vision-model",
        base_url="https://example.invalid/v1",
        max_new_tokens=123,
    )
    response = client.generate(
        prompt="Choose the functional marker.",
        task_instruction="hit the block",
        context_id="unit_context",
        system_prompt="Return JSON only.",
    )

    chat_payload = json.loads(captured[0].decode("utf-8"))
    responses_payload = json.loads(captured[1].decode("utf-8"))
    assert chat_payload["messages"][1]["content"] == "Choose the functional marker."
    assert responses_payload["input"][0]["content"][0] == {
        "type": "input_text",
        "text": "Choose the functional marker.",
    }
    assert response.raw_text == '{"ok": true}'
    assert response.metadata["endpoint"] == "responses"
