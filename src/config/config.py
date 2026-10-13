"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_FOLDER / ".env")


class Config:
    """Configuration class for RAG system"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Model Configuration
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

    # Document Processing
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 120

    # Retrieval
    TOP_K = 4                 # chunks returned by each search
    CHAT_HISTORY_TURNS = 3    # earlier questions the agent remembers
    KEYWORD_WEIGHT = 0.5      # hybrid search: keyword share (meaning search gets the rest)

    # Agent
    AGENT_STEP_LIMIT = 100    # max steps per question (the PII middleware adds steps to every LLM call)
    WEB_RESULTS = 3     

    # Folders
    IRDAI_PDF_FOLDER = PROJECT_FOLDER / "data" / "irdai"
    IRDAI_INDEX_FOLDER = PROJECT_FOLDER / "indexes" / "irdai"

    # Web search: official websites are tried first
    TRUSTED_WEBSITES = ["irdai.gov.in", "policyholder.gov.in", "cioins.co.in"]

    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        return ChatGroq(
            model=cls.LLM_MODEL,
            api_key=cls.GROQ_API_KEY,
            temperature=0,
            max_retries=2,timeout=60
        )
