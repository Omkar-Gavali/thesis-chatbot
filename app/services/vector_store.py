import faiss
import numpy as np
from app.core.logging import logger

class VectorStore:
    def __init__(self, dimension):
        logger.info(f"Initializing FAISS vector store with dimension {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add_documents(self, embeddings, documents):
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.index.add(np.array(embeddings))
        self.documents.extend(documents)

    def search(self, query_embedding, k=5):
        logger.info(f"Searching vector store for top {k} results")
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in indices[0]]
