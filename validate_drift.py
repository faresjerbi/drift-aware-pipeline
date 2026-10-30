
import numpy as np
from drift_detector import (
    load_linked_embeddings,
    cosine_similarity,
    BASELINE_SIZE,
    MINIMUM_RECENT_SIZE,
    THRESHOLD,
)


def main():
    embeddings, interaction_ids = load_linked_embeddings()

    if len(embeddings) < BASELINE_SIZE + MINIMUM_RECENT_SIZE:
        print("Not enough embeddings to validate drift.")
        return

    baseline = embeddings[:BASELINE_SIZE]
    recent = embeddings[-MINIMUM_RECENT_SIZE:]

    baseline_centroid = np.mean(baseline, axis=0)
    recent_centroid = np.mean(recent, axis=0)

    similarity = cosine_similarity(
        baseline_centroid, recent_centroid
    )
    drift_score = 1.0 - similarity

    print(f"Total embeddings: {len(embeddings)}")
    print(
        f"Baseline IDs: {interaction_ids[0]}–"
        f"{interaction_ids[BASELINE_SIZE - 1]}"
    )
    print(
        f"Recent IDs: {interaction_ids[-MINIMUM_RECENT_SIZE]}–"
        f"{interaction_ids[-1]}"
    )
    print(f"Cosine similarity: {similarity:.4f}")
    print(f"Drift score: {drift_score:.4f}")
    print(f"Current threshold: {THRESHOLD:.4f}")
    print(f"Would flag drift: {drift_score > THRESHOLD}")


if __name__ == "__main__":
    main()
