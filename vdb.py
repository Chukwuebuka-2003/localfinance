import os
import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models import Transaction

# Connection to Qdrant Vector DB
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "transactions"

# Embedding provider configuration
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama").lower()  # "ollama" or "lmstudio"

# Ollama configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text-v2-moe")

# LM Studio / OpenAI-compatible configuration for embeddings
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://host.docker.internal:1234")
LMSTUDIO_EMBEDDING_MODEL = os.environ.get("LMSTUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v2-moe")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

# Vector size configuration (depends on embedding model)
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "768"))  # nomic-embed-text-v2-moe is 768

# Initialize clients
ollama_client = None
openai_client = None

if EMBEDDING_PROVIDER == "ollama":
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        print(f"Using Ollama for embeddings at {OLLAMA_HOST} with model {OLLAMA_EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Warning: Could not connect to Ollama for embeddings: {e}")
elif EMBEDDING_PROVIDER == "lmstudio":
    try:
        from openai import OpenAI
        openai_client = OpenAI(
            base_url=f"{LMSTUDIO_HOST}/v1",
            api_key=LMSTUDIO_API_KEY
        )
        print(f"Using LM Studio for embeddings at {LMSTUDIO_HOST} with model {LMSTUDIO_EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client for LM Studio embeddings: {e}")
else:
    print(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. Supported: 'ollama', 'lmstudio'")

# Initialize Qdrant
try:
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
except Exception as e:
    print(f"Warning: Could not connect to Qdrant during init: {e}")
    qdrant_client = None


def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text using the configured provider."""
    if EMBEDDING_PROVIDER == "ollama":
        return _get_embedding_ollama(text)
    elif EMBEDDING_PROVIDER == "lmstudio":
        return _get_embedding_lmstudio(text)
    else:
        print(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")
        return []


def _get_embedding_ollama(text: str) -> List[float]:
    """Generate an embedding using Ollama."""
    if ollama_client is None:
        print("Ollama client not initialized")
        return []

    try:
        response = ollama_client.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding with Ollama: {e}")
        return []


def _get_embedding_lmstudio(text: str) -> List[float]:
    """Generate an embedding using LM Studio / OpenAI-compatible API."""
    if openai_client is None:
        print("LM Studio client not initialized for embeddings")
        return []

    try:
        response = openai_client.embeddings.create(
            model=LMSTUDIO_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding with LM Studio: {e}")
        # LM Studio may not support embeddings, fall back to empty list
        return []


def store_transactions_in_vdb(transactions: List[Transaction]):
    """Embed and store a list of transactions in Qdrant."""
    if not transactions:
        return

    if qdrant_client is None:
        print("Qdrant client not initialized, cannot store transactions")
        return

    points = []
    for tx in transactions:
        tx_string = tx.to_document_string()
        vector = get_embedding(tx_string)

        if not vector:
            print(f"Failed to generate embedding for transaction: {tx_string[:100]}...")
            continue

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=tx.model_dump()
        ))

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"Stored {len(points)} transactions in Qdrant")


def clear_vdb():
    """Wipes all transactions from the vector database by recreating the collection."""
    if qdrant_client is None:
        print("Qdrant client not initialized")
        return False

    try:
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Cleared and recreated Qdrant collection '{COLLECTION_NAME}'")
        return True
    except Exception as e:
        print(f"Failed to clear Qdrant collection: {e}")
        return False


def query_transactions(query: str, limit: int = None) -> List[Transaction]:
    """Retrieve relevant transactions based on a semantic query."""
    if qdrant_client is None:
        print("Qdrant client not initialized")
        return []

    query_vector = get_embedding(query)

    if not query_vector:
        return []

    # Use a high default limit (1000) to effectively return "all" matching results
    # Qdrant requires a limit, so we use a reasonable maximum for personal finance data
    search_limit = limit if limit is not None else 1000

    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=search_limit
    )

    return [Transaction(**hit.payload) for hit in search_result]


def get_all_transactions() -> List[Transaction]:
    """A helper to fetch all stored transactions (up to a limit) for pure analytical tools."""
    if qdrant_client is None:
        return []

    try:
        results, next_page = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000  # Reasonable limit for a personal statement
        )
        return [Transaction(**hit.payload) for hit in results]
    except Exception as e:
        print(f"Error fetching all transactions: {e}")
        return []
