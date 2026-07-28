"""
llm_caller.py — provider-agnostic wrapper around LLM APIs.

Currently implements Gemini (free tier). Claude and OpenAI stubs are in place
for later expansion when API credits are available.

Every call is logged to results/llm_calls/{provider}_{YYYYMMDD}.jsonl for
downstream experiment analysis.
"""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    provider: str
    model: str
    prompt: str
    response_text: str
    latency_seconds: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    timestamp: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseLLMCaller(ABC):
    """Base class handling rate limiting, retries, and logging.

    Subclasses implement provider_name and _make_request().
    """

    provider_name: str = "base"

    def __init__(
        self,
        model: str,
        rpm: int,
        log_dir: Path = Path("results/llm_calls"),
    ):
        self.model = model
        self.rpm = rpm
        self._call_timestamps: deque = deque()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        self.log_file = self.log_dir / f"{self.provider_name}_{stamp}.jsonl"

    @abstractmethod
    def _make_request(self, prompt: str, **kwargs):
        """Return (response_text, input_tokens, output_tokens)."""
        ...

    def _wait_for_rate_limit(self):
        now = time.monotonic()
        while self._call_timestamps and now - self._call_timestamps[0] > 60:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self.rpm:
            wait = 60 - (now - self._call_timestamps[0]) + 0.5
            logger.info(f"[{self.provider_name}] rate limit near cap, sleeping {wait:.1f}s")
            time.sleep(wait)
            self._call_timestamps.popleft()

    def call(self, prompt: str, max_retries: int = 5, **kwargs) -> LLMResponse:
        self._wait_for_rate_limit()
        last_error = None
        start = time.monotonic()

        for attempt in range(max_retries):
            try:
                text, in_tok, out_tok = self._make_request(prompt, **kwargs)
                latency = time.monotonic() - start
                self._call_timestamps.append(time.monotonic())

                response = LLMResponse(
                    provider=self.provider_name,
                    model=self.model,
                    prompt=prompt,
                    response_text=text,
                    latency_seconds=latency,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._log(response)
                return response

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit:
                    wait = min(60, (2 ** attempt) + 1)
                    logger.warning(f"[{self.provider_name}] 429 (attempt {attempt+1}/{max_retries}), waiting {wait}s")
                    time.sleep(wait)
                    continue
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"[{self.provider_name}] error: {e} — retrying in {wait}s")
                    time.sleep(wait)
                    continue

        # All retries exhausted — log and return the failure
        response = LLMResponse(
            provider=self.provider_name,
            model=self.model,
            prompt=prompt,
            response_text="",
            latency_seconds=time.monotonic() - start,
            input_tokens=None,
            output_tokens=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=str(last_error),
        )
        self._log(response)
        return response

    def _log(self, response: LLMResponse):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(asdict(response)) + "\n")


class GeminiCaller(BaseLLMCaller):
    provider_name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash", rpm: int = 10, **kwargs):
        # pyrefly: ignore [missing-import]
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set — check your .env file")
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)
        super().__init__(model=model, rpm=rpm, **kwargs)

    def _make_request(self, prompt: str, **kwargs):
        result = self._client.generate_content(prompt)
        text = result.text
        in_tok = out_tok = None
        if getattr(result, "usage_metadata", None):
            in_tok = result.usage_metadata.prompt_token_count
            out_tok = result.usage_metadata.candidates_token_count
        return text, in_tok, out_tok


class ClaudeCaller(BaseLLMCaller):
    provider_name = "claude"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Add when Anthropic credits are available")

    def _make_request(self, prompt: str, **kwargs):
        raise NotImplementedError


class OpenAICaller(BaseLLMCaller):
    provider_name = "openai"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Add when OpenAI credits are available")

    def _make_request(self, prompt: str, **kwargs):
        raise NotImplementedError


def get_caller(provider: str = "gemini", **kwargs) -> BaseLLMCaller:
    """Factory: create a caller by provider name."""
    providers = {
        "gemini": GeminiCaller,
        "claude": ClaudeCaller,
        "openai": OpenAICaller,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(providers)}")
    return providers[provider](**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    caller = get_caller("gemini")
    response = caller.call("Say hello in one sentence.")
    print(f"Response: {response.response_text}")
    print(f"Latency: {response.latency_seconds:.2f}s")
    print(f"Tokens: in={response.input_tokens}, out={response.output_tokens}")
    print(f"Logged to: {caller.log_file}")