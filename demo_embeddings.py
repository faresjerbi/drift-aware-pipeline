
import requests

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": "What is data drift in machine learning?"
    },
    timeout=60,
)

response.raise_for_status()
data = response.json()
embedding = data["embeddings"][0]

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First 5 values:", embedding[:5])
