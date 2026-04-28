from pathlib import Path
import pdfplumber


def extract_text(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    pages: list[str] = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts: list[str] = []

            # Tables
            for table in page.extract_tables():
                rows = []
                for row in table:
                    if row and any(cell for cell in row):
                        rows.append(" | ".join(str(cell or "").strip() for cell in row))
                if rows:
                    parts.append("\n".join(rows))

            # Full page text
            text = page.extract_text()
            if text:
                parts.append(text)

            pages.append("\n".join(parts))

    return "\n\n--- PAGE BREAK ---\n\n".join(pages)
