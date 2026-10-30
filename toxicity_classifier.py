
from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"

_classifier = None


def get_classifier():
    global _classifier

    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            top_k=None,
        )

    return _classifier


def classify_toxicity(text: str) -> dict:
    """Return toxicity scores and a threshold-based flag."""
    if not text or not text.strip():
        return {
            "toxicity_score": 0.0,
            "toxicity_flag": False,
        }

    predictions = get_classifier()(text[:512])[0]

    scores = {
        item["label"].lower(): float(item["score"])
        for item in predictions
    }

    # The model's toxic label is expected to be "toxic".
    toxicity_score = scores.get("toxic", 0.0)

    return {
        "toxicity_score": toxicity_score,
        "toxicity_flag": toxicity_score >= 0.5,
    }



if __name__ == "__main__":
    examples = [
        "Thank you for your help.",
        "You are an idiot and I hate you.",
        "I strongly disagree with your opinion.",
        "That was a stupid decision.",
        "I will hurt you.",
        "The movie contains offensive language.",
        "You are not very good at this task.",
    ]

    for example in examples:
        result = classify_toxicity(example)
        print(f"\nText: {example}")
        print(f"Result: {result}")


   