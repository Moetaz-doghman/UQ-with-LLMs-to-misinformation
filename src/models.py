from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
GEMINI_GENERATE_CONTENT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_name: str
    api_env_var: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt-4.1-mini": ModelSpec(
        provider="openai",
        model_name="gpt-4.1-mini",
        api_env_var="OPENAI_API_KEY",
    ),
    "claude-3-haiku-20240307": ModelSpec(
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        api_env_var="ANTHROPIC_API_KEY",
    ),
    "gemini-1.5-flash": ModelSpec(
        provider="google",
        model_name="gemini-1.5-flash",
        api_env_var="GOOGLE_API_KEY",
    ),
}


class ModelClient:
    def __init__(self, model_id: str, temperature: float, timeout_seconds: int) -> None:
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unsupported model: {model_id}")
        self.spec = MODEL_SPECS[model_id]
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def _require_api_key(self) -> str:
        api_key = os.getenv(self.spec.api_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Expected environment variable {self.spec.api_env_var}."
            )
        return api_key

    def generate(self, prompt: str) -> str:
        if self.spec.provider == "openai":
            return self._generate_openai(prompt)
        if self.spec.provider == "anthropic":
            return self._generate_anthropic(prompt)
        if self.spec.provider == "google":
            return self._generate_gemini(prompt)
        raise ValueError(f"Unsupported provider: {self.spec.provider}")

    def _post_json(self, url: str, headers: Dict[str, str], payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc

    def _generate_openai(self, prompt: str) -> str:
        payload = {
            "model": self.spec.model_name,
            "input": prompt,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._require_api_key()}",
            "Content-Type": "application/json",
        }
        response = self._post_json(OPENAI_RESPONSES_URL, headers, payload)
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output = response.get("output", [])
        parts = []
        for item in output:
            for content_item in item.get("content", []):
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text", "")
                    if text:
                        parts.append(text)
        if parts:
            return "".join(parts).strip()
        raise RuntimeError("OpenAI response did not contain output_text.")

    def _generate_anthropic(self, prompt: str) -> str:
        payload = {
            "model": self.spec.model_name,
            "max_tokens": 300,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._require_api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        response = self._post_json(ANTHROPIC_MESSAGES_URL, headers, payload)
        content = response.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        text = "".join(parts).strip()
        if text:
            return text
        raise RuntimeError("Anthropic response did not contain text content.")

    def _generate_gemini(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        headers = {
            "x-goog-api-key": self._require_api_key(),
            "Content-Type": "application/json",
        }
        response = self._post_json(
            GEMINI_GENERATE_CONTENT_URL.format(model=self.spec.model_name),
            headers,
            payload,
        )
        candidates = response.get("candidates", [])
        parts = []
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text", "")
                if text:
                    parts.append(text)
        text = "".join(parts).strip()
        if text:
            return text
        raise RuntimeError("Gemini response did not contain text content.")
