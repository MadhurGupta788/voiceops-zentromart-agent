import re
import time
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client()
skip = ("image", "tts", "live", "audio", "embed", "imagen", "veo", "aqa", "robotics", "computer")
allm = list(client.models.list())
models = [m.name.replace("models/", "") for m in allm
          if "generateContent" in (getattr(m, "supported_actions", None) or [])]
if not models:
    models = [m.name.replace("models/", "") for m in allm]
models = [m for m in models if not any(s in m for s in skip)][:25]
print(len(models), "candidates\n")

for n in models:
    t0 = time.time()
    try:
        client.models.generate_content(model=n, contents="Say hi.")
        print(f"OK    {n:42s} {time.time() - t0:4.1f}s", flush=True)
    except Exception as e:
        s = str(e)
        code = re.search(r"\b([45]\d\d)\b", s)
        lim = re.search(r"limit: (\d+)", s)
        per = "PerDay" if "PerDay" in s else ("PerMinute" if "PerMinute" in s else "")
        print(f"FAIL  {n:42s} {code.group(1) if code else '?'} {per} limit={lim.group(1) if lim else '?'}", flush=True)
    time.sleep(1)
