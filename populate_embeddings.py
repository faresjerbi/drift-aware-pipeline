
from embedding_service import generate_embedding, save_embedding

sample_texts = [
    "The customer cannot log into the application.",
    "The website returns an error when I sign in.",
    "Users are experiencing authentication problems.",
    "The application is running slowly today.",
    "The server response time has increased.",
    "The API is returning internal server errors.",
    "The database connection keeps failing.",
    "The service is unavailable for some users.",
    "The application crashes during checkout.",
    "The login page is not loading correctly.",
    "Machine learning models can experience data drift.",
    "The distribution of incoming data has changed.",
    "Model predictions are becoming less accurate.",
    "Recent user queries differ from historical queries.",
    "The model needs monitoring for performance changes.",
    "New data patterns may affect prediction quality.",
    "The input features have different statistical properties.",
    "The system should detect changes in incoming requests.",
    "The model's behavior should be evaluated regularly.",
    "A retraining pipeline can respond to detected drift.",
]

for index, text in enumerate(sample_texts, start=1):
    try:
        embedding = generate_embedding(text)
        save_embedding(text, embedding)
        print(f"{index}/{len(sample_texts)} completed")
    except Exception as error:
        print(f"Failed on item {index}: {error}")

print("Finished generating sample embeddings.")
