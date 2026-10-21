import time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client()
names = [m.name.replace("models/", "") for m in client.models.list() if "flash" in m.name.lower()]
names = [n for n in names if not any(x in n for x in ("image", "tts", "live", "audio", "embed"))][:6]
print("Testing:", names)

settings = {"default": None, "think_off": types.ThinkingConfig(thinking_budget=0)}
for n in names:
    for label, tc in settings.items():
        kw = {"temperature": 0.0, "response_mime_type": "application/json"}
        if tc:
            kw["thinking_config"] = tc
        t0 = time.time()
        try:
            client.models.generate_content(
                model=n,
                contents='Reply as JSON: {"reply": "one short greeting"}',
                config=types.GenerateContentConfig(**kw))
            print(f"{n:35s} {label:10s} {time.time() - t0:5.1f}s", flush=True)
        except Exception as e:
            print(f"{n:35s} {label:10s} FAILED {str(e)[:80]}", flush=True)
