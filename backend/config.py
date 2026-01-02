import os

# --- Configurations for data directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PHAPDIEN_DIR = os.path.join(DATA_DIR, "phap_dien")
VBQPPL_DIR = os.path.join(DATA_DIR, "vbqppl")

# --- Configurations for Database ---
DB_ECHO = False

# --- Configurations for Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# --- Configurations for Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Configurations for Embedding ---
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
EMBEDDING_DIM = 768