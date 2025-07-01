from pathlib import Path
from typing import Optional


def extract_and_clean_data(file_path: str) -> Optional[str]:
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        cleaned_text = text.strip()
        cleaned_text = " ".join(cleaned_text.split())

        return cleaned_text

    except (FileNotFoundError, UnicodeDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")
        return None
