from app.services.document_processor import process_pdf, chunk_text
from app.services.embedding_service import get_embedding_model
from app.services.vector_store import VectorStore
from app.core.config import settings
from app.core.logging import logger
from langchain_groq import ChatGroq
import os

class RAGService:
    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.llm = None

    async def initialize(self):
        logger.info("Initializing RAG system")
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
        logger.info("RAG system initialized successfully")

    async def query(self, query: str):
        logger.info(f"Processing query: {query}")
        query_embedding = self.embedding_model.encode([query])[0]
        relevant_docs = self.vector_store.search(query_embedding)
        
        context = "\n".join(relevant_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        response = self.llm.invoke(prompt)
        return response

rag_service = RAGService()
