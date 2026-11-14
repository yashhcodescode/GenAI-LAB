from abc import ABC, abstractmethod
import os
from google import genai
import requests

# Common interface
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt):
        pass

# Gemini cloud provider
class GeminiProvider(LLMProvider):
    def __init__(self, api_key, model=None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if self.model != "gemini-3.5-flash-lite":
                response = self.client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )
                return response.text
            raise

# Ollama local provider
class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

# Common client
class LLMClient:
    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)
