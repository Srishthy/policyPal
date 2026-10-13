"""RAG state definition for LangGraph"""

from typing import List

from pydantic import BaseModel
from langchain_core.documents import Document


class RAGState(BaseModel):
    """State object for RAG workflow"""

    question: str
    chat_history: List[dict] = []     
    retrieved_docs: List[Document] = []
    answer: str = ""
    sources: List[dict] = []          
    calculations: List[str] = []     
