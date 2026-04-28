import json
import re
from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    model_id: str  # set by each model sibclass

    def extract(self, pdf_text: str, prompt: str) -> list[dict]:
        raw = self._call_api(pdf_text, prompt)
        return self._parse_response(raw)

    @abstractmethod
    def _call_api(self, pdf_text: str, prompt: str) -> str:
        ...

    def _parse_response(self, raw: str) -> list[dict]:
        text = raw.strip()

        # Remove markdown
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        # Return an empty array if the string is not JSON
        if not text or text[0] not in ("[", "{"):
            return []

        return json.loads(text)
