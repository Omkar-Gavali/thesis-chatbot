import os
import pickle
from app.core.config import settings
from app.core.logging import logger
from app.services.document_processor import process_pdf, chunk_text
from app.services.embedding_service import get_embedding_model
from app.services.vector_store import VectorStore
from langchain_groq import ChatGroq

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.llm = ChatGroq(api_key=settings.groq_api_key,model_name=settings.llm_model)

    async def initialize(self):
        logger.info("Initializing RAG system")
        
        if settings.caching_enabled and self._load_cache():
            logger.info("Loaded system from cache")
            return

        text = process_pdf()
        chunks = chunk_text(text)
        
        self.embedding_model = get_embedding_model()
        embeddings = self.embedding_model.encode(chunks)
        
        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.add_documents(embeddings, chunks)
        
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=settings.llm_model
        )

        if settings.caching_enabled:
            self._save_cache()

        logger.info("RAG system initialized successfully")

    def _load_cache(self):
        cache_file = os.path.join(settings.cache_dir, "rag_cache.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            self.vector_store = cached_data["vector_store"]
            self.embedding_model = cached_data["embedding_model"]
            self.llm = cached_data["llm"]
            return True
        return False

    def _save_cache(self):
        os.makedirs(settings.cache_dir, exist_ok=True)
        cache_file = os.path.join(settings.cache_dir, "rag_cache.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "vector_store": self.vector_store,
                "embedding_model": self.embedding_model,
                "llm": self.llm
            }, f)

    async def query(self, query: str):
        logger.info(f"Processing query: {query}")
        query_embedding = self.embedding_model.encode([query])[0]
        relevant_docs = self.vector_store.search(query_embedding)
        
        context = "\n".join(relevant_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = self.llm.invoke(prompt)
        return response

rag_service = RAGService()
