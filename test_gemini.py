from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client()
try:
    r = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say hello in one short sentence.",
    )
    print(r.text)
except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("CODE:", getattr(e, "code", None))
    print("MESSAGE:", str(e)[:500])
