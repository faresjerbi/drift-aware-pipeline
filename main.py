
"""
Drift-Aware Retraining Pipeline
Main entry point.
"""

from dotenv import load_dotenv
from db import test_connection


def main():
    """Initialize the pipeline and verify its database."""

    print("=" * 50)
    print("Drift-Aware Retraining Pipeline")
    print("=" * 50)

    # Load configuration from .env
    load_dotenv()

    print("\n[1/1] Checking database connection...")

    # Verify the database connection and required tables
    database_ready = test_connection()

    if not database_ready:
        print("\nInitialization failed.")
        print("Check your DATABASE_URL and database tables.")
        return

    print("\nInitialization completed successfully.")
    print("The pipeline is ready for the next development phase.")


if __name__ == "__main__":
    main()
