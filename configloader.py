from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()  # Looks for .env in the current working directory

    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX")
    }
