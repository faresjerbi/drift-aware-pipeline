
import os

import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """Connect to the Supabase PostgreSQL database."""
    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL is missing. Check your .env file."
        )

    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def test_connection():
    """Verify the connection and check the required tables."""
    required_tables = {
        "embeddings_log",
        "interaction_log",
        "drift_events",
        "model_versions",
    }

    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                database_name = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name IN (
                          'embeddings_log',
                          'interaction_log',
                          'drift_events',
                          'model_versions'
                      );
                """)

                existing_tables = {
                    row[0] for row in cursor.fetchall()
                }

                print(f"Connected successfully to: {database_name}")

                for table in sorted(required_tables):
                    status = (
                        "OK" if table in existing_tables
                        else "MISSING"
                    )
                    print(f"{status}: {table}")

                return existing_tables == required_tables

    except Exception as error:
        print(f"Database connection failed: {error}")
        return False


if __name__ == "__main__":
    success = test_connection()
    if not success:
        raise SystemExit(1)
