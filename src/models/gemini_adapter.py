import os
import google.generativeai as genai
from .base import LLMAdapter


class GeminiAdapter(LLMAdapter):

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model_id = model
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=None,  # full_prompt
        )

    def _call_api(self, pdf_text: str, prompt: str) -> str:
        full_prompt = f"{prompt}\n\n---\n\n{pdf_text}"
        response = self._model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(temperature=0),
        )
        return response.text
