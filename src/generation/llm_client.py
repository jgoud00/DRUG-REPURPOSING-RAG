"""LLM clients for generation with citation support."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM with metadata."""
    text: str
    model: str
    usage: Optional[dict] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return model name for tracking."""
        pass


class OllamaClient(LLMClient):
    """Client for local Ollama server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.settings = get_settings()
        self.base_url = base_url or self.settings.generation.llm.base_url
        self.model = model or self.settings.generation.llm.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise RuntimeError("ollama package not installed. Run: pip install ollama")
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                    "top_p": kwargs.get("top_p", 0.9),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                },
            )
            text = response["message"]["content"]
            return LLMResponse(text=text, model=self.model, usage=response.get("usage"))
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def model_name(self) -> str:
        return f"ollama:{self.model}"


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible API (fallback)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.settings = get_settings()
        fallback = self.settings.generation.llm.openai_fallback
        self.api_key = api_key or os.getenv(fallback.get("api_key_env", "OPENAI_API_KEY"))
        self.base_url = base_url or fallback.get("base_url", "https://api.openai.com/v1")
        self.model = model or fallback.get("model", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if not self.api_key:
            logger.warning("OpenAI API key not set; fallback will fail")

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 0.9),
                frequency_penalty=min(max((kwargs.get("repeat_penalty", 1.1) - 1.0) * 2, -2.0), 2.0),
            )
            text = response.choices[0].message.content
            return LLMResponse(
                text=text,
                model=self.model,
                usage={"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
                if response.usage else None,
            )
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    def model_name(self) -> str:
        return f"openai:{self.model}"


class MockLLMClient(LLMClient):
    """Deterministic mock client for testing without external dependencies."""

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response or self._default_response()

    def _default_response(self) -> str:
        return (
            "**Drug Candidates:**\n"
            "- **Aspirin (CHEMBL25)**: inhibits COX-1 (P23219) via irreversible acetylation of serine 530, "
            "implicated in Myocardial Infarction [CHEMBL: CHEMBL25]\n"
            "- **Sildenafil (CHEMBL132)**: inhibits PDE5A (O76083) via competitive inhibition, "
            "implicated in Alzheimer's Disease [CHEMBL: CHEMBL132]\n"
            "- **Metformin (CHEMBL1100)**: activates AMPK (Q13131) via LKB1, "
            "implicated in Alzheimer's Disease [CHEMBL: CHEMBL1100]\n\n"
            "**Summary:** Aspirin, Sildenafil, and Metformin show repurposing potential based on "
            "shared pathways with the query disease [PMID: 38412345]."
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> LLMResponse:
        return LLMResponse(text=self.canned_response, model="mock", usage={"prompt_tokens": 0, "completion_tokens": 0})

    def model_name(self) -> str:
        return "mock"


def create_llm_client(
    provider: Optional[str] = None,
    use_mock: bool = False,
    settings=None,
) -> LLMClient:
    """Factory function to create LLM client."""
    if settings is None:
        settings = get_settings()

    if use_mock:
        logger.info("Using MockLLMClient for testing")
        return MockLLMClient()

    provider = provider or settings.generation.llm.provider

    if provider == "ollama":
        logger.info("Using OllamaClient")
        return OllamaClient()
    elif provider == "openai":
        logger.info("Using OpenAIClient (fallback)")
        return OpenAIClient()
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to OllamaClient")
        return OllamaClient()