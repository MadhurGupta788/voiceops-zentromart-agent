from dotenv import load_dotenv
load_dotenv()
import os
from google import genai

client = genai.Client()   # keep a reference so it isn't closed
try:
    r = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL"), contents="Say hi.")
    print("OK:", r.text)
except Exception as e:
    print(str(e)[:1500])
