
import requests
from db import get_connection

OLLAMA_API = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def generate_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_API,
        json={"model": MODEL, "input": text},
        timeout=60,
    )
    response.raise_for_status()

    embedding = response.json()["embeddings"][0]

    if len(embedding) != 768:
        raise ValueError(
            f"Expected 768 dimensions, received {len(embedding)}"
        )

    return embedding


def save_embedding(text: str, embedding: list[float]) -> None:
    # pgvector accepts a vector formatted as a string: [0.1, 0.2, ...]
    vector_string = "[" + ",".join(map(str, embedding)) + "]"

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.embeddings_log (embedding, type)
                VALUES (%s::vector, %s)
                """,
                (vector_string, "query"),
            )

    print("Embedding saved to Supabase!")


def main():
    text = input("Enter text to embed: ").strip()

    if not text:
        print("Please enter some text.")
        return

    embedding = generate_embedding(text)
    print(f"Generated embedding with {len(embedding)} dimensions.")

    save_embedding(text, embedding)


if __name__ == "__main__":
    main()
