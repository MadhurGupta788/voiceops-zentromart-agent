from dotenv import load_dotenv
load_dotenv()
from gnani.tts import GnaniTTSClient

GnaniTTSClient().synthesize(
    "Hello, I am the ZentroMart AI assistant.",
    voice="Kaveri", model="timbre-v2.5",
    language="en-IN", output_file="test.wav",
)
print("ok")
