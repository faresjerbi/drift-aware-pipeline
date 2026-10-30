
import os
from dotenv import load_dotenv
from openai import OpenAI
from db import get_connection
from embedding_service import generate_embedding
from toxicity_classifier import classify_toxicity

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL = "gpt-oss:20b-cloud"

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://ollama.com/v1",
    api_key=OLLAMA_API_KEY,
)


def is_refusal(response_text):
    """Heuristic detection of common refusal wording."""
    phrases = [
        "i cannot help with that",
        "i can't help with that",
        "i am unable to help with that",
        "i'm unable to help with that",
        "i cannot assist with that",
        "i can't assist with that",
        "i'm not able to provide that",
        "i am not able to provide that",
        "i cannot provide that",
        "i can't provide that",
        "i'm sorry, but i cannot",
        "i'm sorry, but i can't",
        "i cannot comply with that request",
        "i can't comply with that request",
    ]

    text = response_text.lower()
    return any(phrase in text for phrase in phrases)


def ask_chatbot(user_query):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_query}],
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else None
    output_tokens = usage.completion_tokens if usage else None
    cost_usd = 0.0  # Placeholder, not verified billing data

    refusal_flag = is_refusal(answer)
    toxicity_result = classify_toxicity(answer)
    toxicity_flag = toxicity_result["toxicity_flag"]

    # Generate an embedding for the user's query.
    embedding = generate_embedding(user_query)
    vector_string = "[" + ",".join(map(str, embedding)) + "]"

    # Save the interaction and its embedding.
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.interaction_log
                    (user_query, model_response, refusal_flag,
                     toxicity_flag, input_tokens, output_tokens, cost_usd)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_query,
                    answer,
                    refusal_flag,
                    toxicity_flag,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                ),
            )
            interaction_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO public.embeddings_log
                    (embedding, type, interaction_id)
                VALUES (%s::vector, %s, %s)
                """,
                (vector_string, "query", interaction_id),
            )

    print(f"Interaction ID: {interaction_id}")
    print(f"Refusal flag: {refusal_flag}")
    print(
        f"Toxicity score: {toxicity_result['toxicity_score']:.4f}"
    )
    print(f"Toxicity flag: {toxicity_flag}")
    print("Interaction and query embedding saved to Supabase.")

    return answer


def main():
    print("Drift-Aware Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        if not user_query:
            continue

        try:
            answer = ask_chatbot(user_query)
            print(f"\nAssistant: {answer}\n")
            print("Interaction saved to Supabase.\n")
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
