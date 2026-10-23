import hashlib
import json
import os
import re
import sys
import time
from google import genai
from google.genai import types
from .config import GEMINI_MODEL, RESULTS_DIR

_client = None
_use_thinking_cfg = True
CACHE_DIR = RESULTS_DIR / ".llm_cache"


class LLMUnavailable(Exception):
    """Raised when the LLM cannot answer (quota, outage). Callers should stop and save progress."""


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(http_options=types.HttpOptions(timeout=60000))
    return _client


def _parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = json.loads(text)
    if isinstance(data, list):
        data = data[0]
    return data


def _config(system):
    global _use_thinking_cfg
    kwargs = dict(system_instruction=system, response_mime_type="application/json", temperature=0.0)
    level = os.getenv("GEMINI_THINKING", "low").strip().lower()
    if _use_thinking_cfg and level not in ("", "default"):
        try:
            if level == "off":
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            else:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        except Exception as e:
            print(f"   [thinking config not usable: {str(e)[:120]}]", file=sys.stderr)
            _use_thinking_cfg = False
    return types.GenerateContentConfig(**kwargs)


def _retry_delay(msg):
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I) or re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", msg)
    return float(m.group(1)) if m else 30.0


def _cache_path(system, user):
    raw = f"{GEMINI_MODEL}\n{os.getenv('GEMINI_THINKING', '')}\n{system}\n{user}"
    return CACHE_DIR / (hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json")


def generate_json(system, user, retries=4):
    """Return a dict from Gemini (from disk cache when possible). Raises LLMUnavailable on quota/outage."""
    global _use_thinking_cfg
    path = _cache_path(system, user)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = _get_client().models.generate_content(
                model=GEMINI_MODEL, contents=user, config=_config(system))
            print(f"   [llm ok in {time.time() - t0:.1f}s]", file=sys.stderr)
            out = _parse_json(r.text)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
        except Exception as e:  # noqa
            last_err = e
            msg = str(e)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            print(f"   [llm attempt {attempt + 1} failed after {time.time() - t0:.1f}s: "
                  f"{type(e).__name__}: {msg[:160]}]", file=sys.stderr)
            if is_429 and "perday" in msg.lower():
                raise LLMUnavailable(f"Daily quota exhausted for {GEMINI_MODEL}.") from e
            if "INVALID_ARGUMENT" in msg or "think" in msg.lower():
                _use_thinking_cfg = False
            wait = min((_retry_delay(msg) + 1) if is_429 else 3 * (attempt + 1), 70)
            print(f"   [waiting {wait:.0f}s before retry]", file=sys.stderr)
            time.sleep(wait)
    raise LLMUnavailable(f"Gave up after {retries} attempts: {str(last_err)[:200]}")
