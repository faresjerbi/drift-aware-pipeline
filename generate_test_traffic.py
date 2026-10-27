
from chatbot import ask_chatbot

test_questions = [
    "What is data drift in machine learning?",
    "Why is my machine learning model becoming less accurate?",
    "How can I monitor model performance?",
    "Why is my API responding slowly?",
    "How do I detect anomalies in incoming data?",
    "What is the difference between data drift and concept drift?",
    "How can I reduce chatbot response time?",
    "What metrics should I track in production?",
    "When should a machine learning model be retrained?",
    "How can I evaluate a new model before deployment?",
    "Why is my database connection failing?",
    "How can I improve chatbot reliability?",
    "What causes changes in incoming data?",
    "How do I monitor prediction quality?",
    "What is an embedding vector?",
    "How can I detect unusual user behavior?",
    "Explain model retraining in simple terms.",
    "How do I build a model monitoring dashboard?",
    "What is cosine similarity used for?",
    "How can I prevent a bad model deployment?",
]

def main():
    print(f"Generating {len(test_questions)} test interactions...\n")

    for index, question in enumerate(test_questions, start=1):
        print(f"[{index}/{len(test_questions)}] {question}")

        try:
            answer = ask_chatbot(question)
            print(f"Saved successfully. Response length: {len(answer)} characters\n")
        except Exception as error:
            print(f"Failed: {error}\n")

    print("Finished generating test traffic.")
    print("Now run: python drift_detector.py")

if __name__ == "__main__":
    main()
