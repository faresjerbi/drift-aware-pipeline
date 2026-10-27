
import numpy as np
from db import get_connection

THRESHOLD = 0.15
BASELINE_SIZE = 20
MINIMUM_RECENT_SIZE = 10
EMBEDDING_DIMENSIONS = 768


def load_linked_embeddings():
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT e.embedding::text, i.id
                FROM public.embeddings_log AS e
                JOIN public.interaction_log AS i
                    ON i.id = e.interaction_id
                WHERE e.type = 'query'
                ORDER BY i.id ASC
            """)
            rows = cursor.fetchall()

    embeddings = []
    interaction_ids = []

    for vector_text, interaction_id in rows:
        values = [
            float(value)
            for value in vector_text.strip("[]").split(",")
        ]

        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Interaction {interaction_id} has "
                f"{len(values)} dimensions, expected "
                f"{EMBEDDING_DIMENSIONS}."
            )

        embeddings.append(values)
        interaction_ids.append(interaction_id)

    if not embeddings:
        return (
            np.empty((0, EMBEDDING_DIMENSIONS)),
            [],
        )

    return np.array(embeddings, dtype=float), interaction_ids


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))



def detect_drift():
    embeddings, interaction_ids = load_linked_embeddings()
    total = len(embeddings)

    print(f"Linked chatbot embeddings: {total}")

    if total < BASELINE_SIZE + MINIMUM_RECENT_SIZE:
        print(
            f"Need at least {BASELINE_SIZE + MINIMUM_RECENT_SIZE} "
            "linked embeddings."
        )
        return None 

    baseline = embeddings[:BASELINE_SIZE]
    recent = embeddings[-MINIMUM_RECENT_SIZE:]
    recent_start_id = interaction_ids[-MINIMUM_RECENT_SIZE]
    recent_end_id = interaction_ids[-1]

    # Identify this exact recent window.
    signal = (
        f"baseline_vs_recent_"
        f"{recent_start_id}_{recent_end_id}"
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 1
                FROM public.drift_events
                WHERE signal = %s
                LIMIT 1
            """, (signal,))
            already_evaluated = cursor.fetchone() is not None

    if already_evaluated:
        print(
            f"Window {recent_start_id}–{recent_end_id} "
            "has already been evaluated. Skipping."
        )
        return None 

    baseline_centroid = np.mean(baseline, axis=0)
    recent_centroid = np.mean(recent, axis=0)

    similarity = cosine_similarity(
        baseline_centroid,
        recent_centroid,
    )
    drift_score = 1.0 - similarity
    drift_detected = drift_score > THRESHOLD
    action = "drift_detected" if drift_detected else "no_drift"

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO public.drift_events
                    (signal, score, threshold, action_taken)
                VALUES (%s, %s, %s, %s)
            """, (signal, drift_score, THRESHOLD, action))
    event_id = cursor.fetchone()[0]
    print(f"Baseline size: {len(baseline)}")
    print(f"Baseline interaction IDs: {interaction_ids[0]}–{interaction_ids[BASELINE_SIZE - 1]}")
    print(f"Recent size: {len(recent)}")
    print(f"Recent interaction IDs: {recent_start_id}–{recent_end_id}")
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Drift score: {drift_score:.4f}")
    print(f"Threshold: {THRESHOLD:.4f}")
    print(f"Drift detected: {drift_detected}")
    print(f"Event saved: {action}")

    print(f"Event saved: {action}")
    return event_id, action

if __name__ == "__main__":
    detect_drift()

