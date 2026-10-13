from typing import List

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnode import RAGNodes


class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, policy_store, rules_store, llm):
        """
        Initialize graph builder

        Args:
            policy_store: VectorStore with the user's policy (None if not uploaded yet)
            rules_store: VectorStore with the IRDAI rules
            llm: Language model instance
        """
        self.nodes = RAGNodes(policy_store, rules_store, llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph

        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # Set entry point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        # Compile graph
        self.graph = builder.compile()
        return self.graph

    def run(self, question: str, chat_history: List[dict] = None) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question
            chat_history: Earlier questions and answers

        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question, chat_history=chat_history or [])
        return self.graph.invoke(initial_state)
