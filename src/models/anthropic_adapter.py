import os
import anthropic
from .base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model_id = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _call_api(self, pdf_text: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=4096,
            system=prompt,
            messages=[{"role": "user", "content": pdf_text}],
        )
        return response.content[0].text
