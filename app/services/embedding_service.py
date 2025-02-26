from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger

def get_embedding_model():
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return SentenceTransformer(settings.embedding_model)
