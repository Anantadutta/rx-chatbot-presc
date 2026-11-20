"""
Reads a handwritten prescription (image or PDF) with Claude's vision model
and returns structured data, with a confidence score and alternative
readings for every field that was hard to read.
"""
import base64
import json
import os
import re
import time

import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

PROMPT = """You are an expert at reading doctors' handwritten medical prescriptions.
Carefully read the attached prescription. Doctors' handwriting is often hard to read,
so use medical context (common drug names, standard dosages, abbreviations such as
OD, BD, TDS, QID, HS, SOS, 1-0-1, Rx, Tab, Cap, Syp) to decide on the most likely reading.

Return ONLY a JSON object, no markdown, in exactly this shape:
{
  "patient":  {"name": str|null, "patient_id": str|null, "age": str|null,
               "gender": str|null, "phone": str|null},
  "doctor":   {"name": str|null, "registration_no": str|null, "clinic": str|null},
  "date": "YYYY-MM-DD" | null,
  "diagnosis": str|null,
  "medicines": [
    {"name": str, "dosage": str|null, "frequency": str|null, "duration": str|null,
     "confidence": float, "alternatives": [str]}
  ],
  "advice": str|null,
  "field_confidence": {"patient_name": float, "patient_id": float,
                       "doctor_name": float, "date": float},
  "illegible_parts": [str]
}

Rules:
- confidence values are between 0 and 1 (1 = certain).
- "alternatives" lists other plausible readings of a medicine name, most likely first.
- Use null for anything not present. Never invent a value you cannot see.
- Expand abbreviations in "frequency" when obvious, e.g. "1-0-1 (BD, twice daily)".
"""

MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".pdf": "application/pdf",
}


def _content_block(file_bytes: bytes, filename: str) -> dict:
    ext = os.path.splitext(filename.lower())[1]
    media_type = MEDIA_TYPES.get(ext)
    if not media_type:
        raise ValueError(f"Unsupported file type '{ext}'. Use JPG, PNG, WEBP, GIF or PDF.")
    data = base64.standard_b64encode(file_bytes).decode()
    kind = "document" if media_type == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


def read_prescription(file_bytes: bytes, filename: str, api_key: str | None = None,
                      provider: str | None = None) -> dict:
    """Read the prescription with Claude or Gemini and return the extracted fields.

    provider: "claude" or "gemini". If not given, it is chosen from which
    API key is set (ANTHROPIC_API_KEY or GEMINI_API_KEY).
    """
    if provider is None:
        provider = "claude" if os.getenv("ANTHROPIC_API_KEY") else "gemini"
    if provider == "gemini":
        return _read_with_gemini(file_bytes, filename, api_key or os.getenv("GEMINI_API_KEY"))
    return _read_with_claude(file_bytes, filename, api_key or os.getenv("ANTHROPIC_API_KEY"))


def _read_with_gemini(file_bytes: bytes, filename: str, api_key: str) -> dict:
    """Google Gemini - has a free tier (get a key at aistudio.google.com)."""
    from google import genai
    from google.genai import types

    ext = os.path.splitext(filename.lower())[1]
    media_type = MEDIA_TYPES.get(ext)
    if not media_type:
        raise ValueError(f"Unsupported file type '{ext}'. Use JPG, PNG, WEBP, GIF or PDF.")
    client = genai.Client(api_key=(api_key or "").strip(),
                          http_options=types.HttpOptions(timeout=300_000))  # 5 min, ms
    contents = [types.Part.from_bytes(data=file_bytes, mime_type=media_type), PROMPT]

    # Try the main model, then lighter fallbacks if Google is slow or busy.
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    last_error = None
    for model in models:
        for attempt in range(2):
            try:
                try:  # low thinking = much faster, still accurate for reading text
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level="low"))
                    response = client.models.generate_content(
                        model=model, contents=contents, config=config)
                except Exception as e:
                    if "thinking" not in str(e).lower():
                        raise
                    response = client.models.generate_content(  # model without thinking
                        model=model, contents=contents,
                        config=types.GenerateContentConfig(response_mime_type="application/json"))
                return _parse_json(response.text)
            except Exception as e:
                last_error = e
                msg = str(e)
                if any(c in msg for c in ("401", "403", "API_KEY_INVALID", "PERMISSION_DENIED")):
                    raise  # bad key: no point retrying
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # model not available: try the next one
                if any(c in msg for c in ("504", "503", "500", "DEADLINE", "UNAVAILABLE",
                                          "timed out", "429", "RESOURCE_EXHAUSTED")):
                    time.sleep(5 * (attempt + 1))
                    continue  # busy / slow: wait and retry
                raise
    raise RuntimeError(f"Gemini is busy or slow right now, tried {', '.join(models)}. "
                       f"Please try again in a minute. Last error: {last_error}")


def _read_with_claude(file_bytes: bytes, filename: str, api_key: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key, timeout=180, max_retries=3)
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [_content_block(file_bytes, filename), {"type": "text", "text": PROMPT}],
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return _parse_json(text)
