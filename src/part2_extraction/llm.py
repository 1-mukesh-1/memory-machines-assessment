"""
LLM Abstraction Layer - Provider-agnostic interface for LLM calls.

Supports: OpenAI, Anthropic, Ollama

Configuration via environment variables:
- LLM_PROVIDER: openai | anthropic | ollama
- LLM_MODEL: model name (e.g., gpt-4o, claude-sonnet-4-20250514, llama3)
- OPENAI_API_KEY / ANTHROPIC_API_KEY: API keys
- OLLAMA_BASE_URL: For ollama (default: http://localhost:11434)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class LLMResponse:
    """Raw LLM response wrapper."""
    content: str
    model: str
    usage: dict = None


class BaseLLM(ABC):
    """Abstract base for LLM providers."""
    
    provider: str
    model: str
    
    @abstractmethod
    async def complete(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        """Send messages and get completion."""
        pass
    
    async def extract_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Get completion and parse as JSON."""
        response = await self.complete(messages, temperature)
        return self._parse_json(response.content)
    
    def _parse_json(self, content: str) -> dict:
        """Extract JSON from response, handling markdown fences and reasoning."""
        import re
        
        # Try to find JSON in code block first (greedy to get full object)
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Find outermost JSON object (greedy)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
        raise ValueError(f"No JSON found in response: {content[:200]}")


class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation."""
    
    provider = "anthropic"
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model
        self._client = None
    
    async def _get_client(self):
        if not self._client:
            try:
                import anthropic
            except ImportError:
                raise ImportError("pip install anthropic")
            self._client = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._client
    
    async def complete(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        client = await self._get_client()
        
        # Separate system message if present
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": chat_messages, "temperature": temperature}
        if system:
            kwargs["system"] = system
        
        response = await client.messages.create(**kwargs)
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
        )


class OpenAILLM(BaseLLM):
    """OpenAI GPT implementation."""
    
    provider = "openai"
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None
    
    async def _get_client(self):
        if not self._client:
            try:
                import openai
            except ImportError:
                raise ImportError("pip install openai")
            self._client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client
    
    async def complete(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        client = await self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
        )


class OllamaLLM(BaseLLM):
    """Ollama local model implementation."""
    
    provider = "ollama"
    
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    async def complete(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        import httpx
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
            )
            response.raise_for_status()
            data = response.json()
        
        return LLMResponse(content=data["message"]["content"], model=self.model)


class MockLLM(BaseLLM):
    """Mock LLM for testing without API credentials."""
    
    provider = "mock"
    model = "mock-v1"
    
    async def complete(self, messages: list[dict], temperature: float = 0.0) -> LLMResponse:
        """Return deterministic fake extraction."""
        # Extract event name from user message
        user_msg = messages[-1]["content"]
        
        # Generate fake but plausible response
        response = {
            "claims": [
                "Lincoln was present during this event",
                "The event had significant political implications",
            ],
            "temporal_details": {"date": "1860s", "time": None},
            "tone": "Neutral"
        }
        
        # If text is very short or no event keywords, return empty
        if len(user_msg) < 500 or "not mentioned" in user_msg.lower():
            response = {"claims": [], "temporal_details": None, "tone": "Neutral"}
        
        return LLMResponse(
            content=json.dumps(response),
            model=self.model,
            usage={"input": 100, "output": 50},
        )


def get_llm(provider: str = None, model: str = None) -> BaseLLM:
    """
    Factory function to get LLM instance.
    
    Priority: arguments > env vars > defaults
    
    Providers: anthropic, openai, ollama, mock
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    
    if provider == "mock":
        return MockLLM()
    elif provider == "anthropic":
        model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        return AnthropicLLM(model)
    elif provider == "openai":
        model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        return OpenAILLM(model)
    elif provider == "ollama":
        model = model or os.environ.get("LLM_MODEL", "llama3")
        return OllamaLLM(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")