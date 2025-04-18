"""
API for LLM clients.

Adapted from https://github.com/lilakk/BooookScore/blob/main/booookscore/utils.py
"""

import time
from pathlib import Path

import cohere
import google.generativeai as genai
from anthropic import Anthropic
from loguru import logger
from openai import OpenAI


class APIClient:
    """Common class for API clients."""

    def __init__(self, api: str, key_path: str, model: str) -> None:
        """Initialize API client."""
        assert key_path.endswith(".txt"), "api key path must be a txt file."
        self.api = api
        self.model = model
        if api == "openai":
            self.client = OpenAIClient(key_path, model)
        elif api == "anthropic":
            self.client = AnthropicClient(key_path, model)
        elif api == "together":
            self.client = TogetherClient(key_path, model)
        elif api == "cohere":
            self.client = CohereClient(key_path, model)
        elif api == "gemini":
            self.client = GeminiClient(key_path, model)
        else:
            msg = f"API {api} not supported, custom implementation required."
            raise ValueError(msg)

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Obtain response from API."""
        return self.client.obtain_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


class BaseClient:
    """Base class for API clients."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize API client."""
        with Path(key_path).open() as f:
            self.key = f.read().strip()
        self.model = model

    def obtain_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        max_attempts: int = 3,
    ) -> str:
        """Obtain response from API."""
        response = None
        num_attempts = 0
        while response is None:
            try:
                response = self.send_request(prompt, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001, PERF203
                logger.warning(e)
                num_attempts += 1
                logger.warning(
                    f"Attempt {num_attempts} failed, trying again after 5 seconds..."
                )
                if num_attempts >= max_attempts:
                    return ""
                time.sleep(5)
        return response

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> None:
        """Send request is implemented by the subclasses."""
        raise NotImplementedError


class OpenAIClient(BaseClient):
    """OpenAI API client."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize OpenAI API client."""
        super().__init__(key_path, model)
        self.client = OpenAI(api_key=self.key)

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Send request to OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient(BaseClient):
    """Anthropic API client."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize Anthropic API client."""
        super().__init__(key_path, model)
        self.client = Anthropic(api_key=self.key)

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Send request to Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class TogetherClient(BaseClient):
    """Together AI API client."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize Together AI API client."""
        super().__init__(key_path, model)
        self.client = OpenAI(api_key=self.key, base_url="https://api.together.xyz/v1")

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Send request to Together AI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class CohereClient(BaseClient):
    """Cohere API client."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize Cohere API client."""
        super().__init__(key_path, model)
        with Path(key_path).open() as f:
            api_key = f.read().strip()
        self.client = cohere.Client(api_key=api_key)

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:  # noqa: ARG002
        """Send request to Cohere API."""
        response = self.client.chat(
            model=self.model,
            message=prompt,
        )
        return response.text


class GeminiClient(BaseClient):
    """Gemini API client."""

    def __init__(self, key_path: str, model: str) -> None:
        """Initialize Gemini API client."""
        super().__init__(key_path, model)
        genai.configure(api_key=self.key)
        self.client = genai.GenerativeModel(model)

    def send_request(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Send request to Gemini API."""
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text
