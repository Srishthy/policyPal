"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_answer"""

import re
from typing import List, Optional

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.config.config import Config
from src.nodes.privacy import pii_middleware
from src.state.rag_state import RAGState
from src.tools.calculators import check_waiting_period, estimate_claim

SYSTEM_PROMPT = """You are PolicyPal. You help people in India understand their health insurance.

There are two kinds of questions:

A. Questions about the user's own policy ("Is cataract covered for me?", "What is my room rent limit?")
   - You get clauses from their policy labelled [P1], [P2]... Answer from these.
   - If they don't answer it, use search_my_policy with other words.
   - If the policy still doesn't say, tell the user to check with their insurer. Never guess what their policy covers.
   - If no policy is uploaded, ask them to upload it.

B. General questions about health insurance ("What is a co-payment?", "How does porting work?")
   - First use search_irdai_rules and answer from what it finds, citing [R1], [R2]...
   - If that doesn't answer it, use search_web and cite [W1], [W2]... Prefer official sources (IRDAI).
   - Never use web results to say what the user's own policy covers. Policies differ.

Rules:
1. Never do date or money calculations yourself. Read the numbers from the policy and use
   check_waiting_period or estimate_claim. If a number is missing, ask the user for it.
   Before estimate_claim, make sure you know the room rent limit AND the co-payment:
   if they are not in the clauses you have, call search_my_policy for them (e.g. "co-payment").
2. Every fact in your answer must come from the text the tools gave you, with its label
   like [P1], [R2] or [W1] right after it. Only use labels you were given.
   If something is not in that text, don't say it. Say you couldn't find it instead.
3. If the policy goes against IRDAI rules, say so. IRDAI rules usually win.
4. For disputes, explain the steps: insurer's grievance cell → IRDAI Bima Bharosa portal → Insurance Ombudsman.
5. ONLY for questions about the user's own policy, start with one verdict:
   Covered / Not covered / Covered with limits / Not clear.
   General questions get a plain answer with no verdict.
6. Text like [REDACTED_PHONE] is personal data we hid on purpose. Don't ask for it.
7. Write plain text. No LaTeX, no formulas and no calculations of your own.
   After using a calculator, give ONLY its final result (e.g. "the insurer pays about ₹77,662, you pay about
   ₹42,338") and say the working is shown under "How we calculated this". Do not retell the steps.
   Only ask for numbers the tools need (estimate_claim does not need the number of days).
8. For follow-up questions, use the earlier conversation.
9. Never say the policy doesn't mention something until you have called search_my_policy
   at least once with different words (e.g. "co-payment", "co-pay percentage claim").

End every answer with: "This is general information, not legal or financial advice."
"""


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, policy_store, rules_store, llm):
        """
        Initialize RAG nodes

        Args:
            policy_store: VectorStore with the user's policy (None if not uploaded yet)
            rules_store: VectorStore with the IRDAI rules (None if not built yet)
            llm: Language model instance
        """
        self.policy_store = policy_store
        self.rules_store = rules_store
        self.llm = llm
        self._agent = None    # lazy-init agent
        self.sources = []     # everything shown to the agent for the current question

        # web search (Tavily): official websites first, then the whole web
        self.trusted_web_search = None
        self.open_web_search = None
        if Config.TAVILY_API_KEY:
            from langchain_tavily import TavilySearch
            self.trusted_web_search = TavilySearch(max_results=Config.WEB_RESULTS, include_domains=Config.TRUSTED_WEBSITES)
            self.open_web_search = TavilySearch(max_results=Config.WEB_RESULTS)

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        self.sources = []  # new question, start a fresh list

        docs = []
        if self.policy_store is not None:
            docs = self.policy_store.retrieve(state.question)

            # a follow-up like "so do that calculation" means little on its own,
            # so we also add 2 results from searching with the previous question included
            if state.chat_history:
                follow_up_text = state.chat_history[-1]["question"] + " " + state.question
                for doc in self.policy_store.retrieve(follow_up_text, k=2):
                    if doc not in docs:
                        docs.append(doc)

        return RAGState(
            question=state.question,
            chat_history=state.chat_history,
            retrieved_docs=docs
        )

    def _add_source(self, letter: str, title: str, text: str, file: str, page=None, clause: str = "") -> str:
        """Give a search result the next label ([P1], [R2], [W1]...) and remember it for the app"""
        number = len([s for s in self.sources if s["label"].startswith(letter)]) + 1
        label = f"{letter}{number}"
        self.sources.append({"label": label, "file": file, "page": page, "clause": clause, "text": text})
        return f"[{label}] {title}\n{text}"

    def _format_docs(self, docs: List[Document], letter: str) -> str:
        """Turn policy or IRDAI chunks into labelled text for the agent"""
        if not docs:
            return "No documents found."

        texts = []
        for doc in docs:
            file = doc.metadata.get("file")
            page = doc.metadata.get("page")
            clause = doc.metadata.get("clause", "")
            title = f"{file}, page {page}, {clause}"
            texts.append(self._add_source(letter, title, doc.page_content, file, page, clause))
        return "\n\n".join(texts)

    def _search_policy(self, query: str) -> str:
        if self.policy_store is None:
            return "No policy uploaded. Ask the user to upload their policy PDF."
        return self._format_docs(self.policy_store.retrieve(query), "P")

    def _search_irdai_rules(self, query: str) -> str:
        if self.rules_store is None:
            return "IRDAI rules are not loaded."
        return self._format_docs(self.rules_store.retrieve(query), "R")

    def _search_web(self, query: str) -> str:
        if self.trusted_web_search is None:
            return "Web search is not set up (no TAVILY_API_KEY)."

        query = f"{query} health insurance India"
        try:
            results = self.trusted_web_search.invoke({"query": query}).get("results", [])
            if not results:
                results = self.open_web_search.invoke({"query": query}).get("results", [])
        except Exception as error:
            return f"Web search failed: {error}"

        if not results:
            return "Nothing found on the web."

        texts = []
        for result in results:
            title = f"{result['title']} ({result['url']})"
            texts.append(self._add_source("W", title, result["content"], file=result["url"], clause=result["title"]))
        return "\n\n".join(texts)

    def _build_tools(self) -> list:
        """Build policy search + IRDAI search + web search + calculator tools"""

        # StructuredTool reads the input name ("query") from the function,
        # so the LLM knows exactly what to send
        policy_tool = StructuredTool.from_function(
            func=self._search_policy,
            name="search_my_policy",
            description="Search the user's own policy: coverage, limits, waiting periods, exclusions, co-payment.",
        )

        rules_tool = StructuredTool.from_function(
            func=self._search_irdai_rules,
            name="search_irdai_rules",
            description="Search IRDAI rules: general health insurance questions, and what insurers must or must not do.",
        )

        web_tool = StructuredTool.from_function(
            func=self._search_web,
            name="search_web",
            description="Search the internet. Only for general questions, and only if search_irdai_rules found nothing.",
        )

        return [policy_tool, rules_tool, web_tool, check_waiting_period, estimate_claim]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        self._agent = create_agent(
            self.llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=pii_middleware(),  # hides Aadhaar, PAN, phone and email before every LLM call
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with policy search, IRDAI search, web search and calculators.
        """
        if self._agent is None:
            self._build_agent()

        # earlier questions and answers go in the same message, because the
        # PII middleware only checks the newest user message
        history = ""
        for turn in state.chat_history[-Config.CHAT_HISTORY_TURNS:]:
            history += f"User: {turn['question']}\nPolicyPal: {turn['answer']}\n\n"

        if self.policy_store is None:
            policy_text = "The user has not uploaded a policy yet."
        else:
            policy_text = self._format_docs(state.retrieved_docs, "P")

        message = (
            f"Earlier conversation:\n{history or '(none)'}\n"
            f"Question: {state.question}\n\n"
            f"Clauses found in the user's policy:\n{policy_text}"
        )

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": Config.AGENT_STEP_LIMIT},  # the PII middleware adds steps to every LLM call
        )

        messages = result.get("messages", [])

        # the calculator results, shown to the user under "How we calculated this"
        calculations = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name in ["check_waiting_period", "estimate_claim"]:
                if msg.content not in calculations:  # show each result only once
                    calculations.append(msg.content)

        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        # some models write citations as 【R1】 instead of [R1]; make them all [R1]
        if answer:
            answer = answer.replace("【", "[").replace("】", "]")

            # tool names are not citations
            for tool_name in ["estimate_claim", "check_waiting_period"]:
                answer = answer.replace(f" [{tool_name}]", "").replace(f"[{tool_name}]", "")

            # remove labels the agent made up (like [P4-2]): keep only labels we actually gave it
            known_labels = [source["label"] for source in self.sources]
            for label in re.findall(r"\[([PRW]\d[^\]]*)\]", answer):
                if label not in known_labels:
                    answer = answer.replace(f" [{label}]", "").replace(f"[{label}]", "")

        return RAGState(
            question=state.question,
            chat_history=state.chat_history,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer.",
            sources=self.sources,
            calculations=calculations,
        )
