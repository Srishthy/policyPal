"""Vector store module for document embedding and retrieval.

Retrieval is hybrid: keyword search (BM25) + meaning search (FAISS), merged by LangChain's EnsembleRetriever.
"""

import re
from pathlib import Path
from typing import List

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.config import Config


def split_into_words(text: str) -> List[str]:
    """Lowercase and keep only letters and numbers, so "Co-payment?" matches "co-payment"."""
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorStore:
    """Manages vector store operations"""

    def __init__(self, embedding=None, model_name: str = Config.EMBEDDING_MODEL):
        # pass an existing embedding model to reuse it (the policy and the IRDAI rules share one)
        if embedding is None:
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        self.embedding = embedding
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document], k: int = Config.TOP_K):
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self._build_hybrid_retriever(documents, k)

    def _build_hybrid_retriever(self, documents: List[Document], k: int):
        # keyword search: good at exact words like "co-payment", "cataract" or "Excl01"
        keyword_retriever = BM25Retriever.from_documents(documents, preprocess_func=split_into_words)
        keyword_retriever.k = k

        # meaning search: good at questions asked in different words
        meaning_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # run both and merge their rankings (Reciprocal Rank Fusion)
        self.retriever = EnsembleRetriever(
            retrievers=[keyword_retriever, meaning_retriever],
            weights=[Config.KEYWORD_WEIGHT, 1 - Config.KEYWORD_WEIGHT],
        )

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = Config.TOP_K) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        # the two searches together can return up to 2k chunks; keep the best k
        return self.retriever.invoke(query)[:k]

    def save(self, folder: Path):
        """Save the index to disk (used for the IRDAI rules, which never change)"""
        self.vectorstore.save_local(str(folder))

    def load(self, folder: Path, k: int = Config.TOP_K) -> bool:
        """Load a saved index. Returns False if it hasn't been built yet."""
        if not (Path(folder) / "index.faiss").exists():
            return False
        # safe here because we only load indexes this project built itself
        self.vectorstore = FAISS.load_local(str(folder), self.embedding, allow_dangerous_deserialization=True)

        # the keyword search isn't saved, so rebuild it from the chunks stored inside FAISS
        documents = list(self.vectorstore.docstore._dict.values())
        self._build_hybrid_retriever(documents, k)
        return True
