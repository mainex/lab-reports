import os
from openai import OpenAI
from .base import LLMAdapter


class OpenAIAdapter(LLMAdapter):

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model_id = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _call_api(self, pdf_text: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": pdf_text},
            ],
        )
        return response.choices[0].message.content
