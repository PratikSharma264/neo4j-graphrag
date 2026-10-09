
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

KNOWLEDGE_BASE_DIR = (
    ROOT_DIR / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
).resolve()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
)
OLLAMA_LLM_MODEL = os.getenv(
    "OLLAMA_LLM_MODEL", "llama3.1:8b"
)
OLLAMA_EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL", "nomic-embed-text"
)

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
EMBEDDING_DIMENSIONS = 768
TOP_K = 5