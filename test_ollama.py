
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

if not api_key:
    raise ValueError("OLLAMA_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="gpt-oss:20b-cloud",
    messages=[
        {"role": "user", "content": "Explain AI model drift in two sentences."}
    ],
)

print(response.choices[0].message.content)
