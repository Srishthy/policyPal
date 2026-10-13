"""Test the whole flow without API keys, using a fake LLM, fake embeddings and fake web search.

    python tests/make_demo_policy.py
    python tests/test_offline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder
from src.tools.calculators import check_waiting_period, estimate_claim
from src.vectorstore.vector_store import VectorStore

sent_to_llm = []
failed = []


def check(description, passed):
    print(("PASS  " if passed else "FAIL  ") + description)
    if not passed:
        failed.append(description)


class FakeLLM(BaseChatModel):
    """Replies with a fixed list of messages, one per call, and records what it was sent."""
    replies: list
    call_number: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self):
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        sent_to_llm.append("\n".join(str(m.content) for m in messages))
        reply = self.replies[self.call_number]
        self.call_number += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def fake_store(documents):
    store = VectorStore(embedding=DeterministicFakeEmbedding(size=64))
    store.create_vectorstore(documents)
    return store


# ---------- 1. calculators ----------
waiting = check_waiting_period.invoke({"cover_start_date": "2023-04-01", "waiting_period_months": 24,
                                       "treatment_date": "2025-11-20"})
print(waiting)
check("cataract waiting period ends 2025-04-01 and is over", "2025-04-01" in waiting and "is over" in waiting)

claim = estimate_claim.invoke({"total_bill": 120000, "medicines_and_tests": 18000, "room_rent_per_day": 8000,
                               "sum_insured": 500000, "room_rent_limit_percent": 1, "copay_percent": 10})
print(claim)
# (120,000 − 18,000) × 62.5% + 18,000 = 81,750; minus 10% co-pay = 73,575
check("claim estimate: insurer pays ₹73,575", "Insurer pays about ₹73,575" in claim)

# Arogya Sanjeevani style limit: 2% of sum insured, maximum ₹5,000 per day → the lower one (₹5,000) applies
claim = estimate_claim.invoke({"total_bill": 120000, "medicines_and_tests": 18000, "room_rent_per_day": 8000,
                               "sum_insured": 500000, "room_rent_limit_percent": 2,
                               "room_rent_limit_per_day": 5000, "copay_percent": 5})
print(claim)
check("'2% but max ₹5,000' uses the lower limit (₹5,000/day)", "limit ₹5,000/day" in claim)

# ---------- 2. reading the PDF ----------
pdf_path = Path(__file__).parent / "demo_policy.pdf"
chunks = DocumentProcessor(chunk_size=300, chunk_overlap=30).process_pdfs([pdf_path])
clauses = {chunk.metadata["clause"] for chunk in chunks}
print("Clauses found:", sorted(clauses))
check("clause '4.2 Room Rent' found", "4.2 Room Rent" in clauses)
check("clause '5.2 Specific Waiting Period' found", "5.2 Specific Waiting Period" in clauses)
check("page numbers start from 1", min(chunk.metadata["page"] for chunk in chunks) == 1)
processor = DocumentProcessor()
check("page footer '261 Page' is not a clause heading", not processor.is_clause_heading("261 Page"))
check("'26 | Page' is not a clause heading", not processor.is_clause_heading("26 | Page"))
check("'25 I P age' (scanned footer) is not a clause heading", not processor.is_clause_heading("25 I P age"))

# ---------- 3. hybrid search (keywords + meaning) ----------
hybrid_store = fake_store(chunks)  # fake embeddings: only the keyword half finds anything useful
top_result = hybrid_store.retrieve("Is there a co-payment?")[0]
print("Top result for 'Is there a co-payment?':", top_result.page_content[:80].replace("\n", " "))
check("keyword search finds the co-payment clause first", "co-payment of 10%" in top_result.page_content.lower())
check("retrieve returns at most TOP_K chunks", len(hybrid_store.retrieve("room rent")) <= 4)

# ---------- 3b. policy questions ----------
from src.nodes.reactnode import RAGNodes
tool_inputs = {t.name: list(t.args) for t in RAGNodes(None, None, None)._build_tools()}
print("Tool inputs:", tool_inputs)
check("search tools take a clear 'query' input (Groq rejects '__arg1')",
      tool_inputs["search_my_policy"] == ["query"] and tool_inputs["search_irdai_rules"] == ["query"]
      and tool_inputs["search_web"] == ["query"])

policy_store = fake_store(chunks)
search_texts = []

def fake_retrieve(query, k=4):
    search_texts.append(query)
    return chunks[:k]  # always include page 1 (it has phone, PAN, email)

policy_store.retrieve = fake_retrieve

llm = FakeLLM(replies=[
    AIMessage(content="", tool_calls=[{"name": "check_waiting_period", "id": "1", "args": {
        "cover_start_date": "2023-04-01", "waiting_period_months": 24, "treatment_date": "2025-11-20"}}]),
    AIMessage(content="Covered. The waiting period ended on 1 Apr 2025 [P1]."),
    AIMessage(content="", tool_calls=[{"name": "estimate_claim", "id": "2", "args": {
        "total_bill": 120000, "medicines_and_tests": 18000, "room_rent_per_day": 8000,
        "sum_insured": 500000, "room_rent_limit_percent": 1, "copay_percent": 10}}]),
    AIMessage(content="Insurance will pay about ₹73,575 [P2] [P4-2] [P99] [estimate_claim]."),
])
graph = GraphBuilder(policy_store, None, llm)
print("\nGraph:", list(graph.build().get_graph().nodes))

result1 = graph.run("My Aadhaar is 2345 6789 0123. Is cataract surgery on 20 Nov 2025 covered?")
print("Answer 1:", result1["answer"])
check("question 1 shows the waiting period calculation", "waiting period ended on 2025-04-01" in result1["calculations"][0])
check("tool result went back to the LLM", "waiting period is over" in sent_to_llm[1])

history = [{"question": "Is cataract covered?", "answer": result1["answer"]}]
result2 = graph.run("My bill is 1.2 lakh, 8000/day room, 4 days. How much will I get?", history)
print("Answer 2:", result2["answer"])
check("question 2 shows the claim calculation", "Insurer pays about ₹73,575" in result2["calculations"][0])
check("claim tool returned ₹73,575 to the LLM", "Insurer pays about ₹73,575" in sent_to_llm[3])
check("made-up labels [P4-2], [P99], [estimate_claim] removed, real [P2] kept",
      "[P2]" in result2["answer"] and "[P4-2]" not in result2["answer"] and "[P99]" not in result2["answer"]
      and "[estimate_claim]" not in result2["answer"])
check("co-payment is a required calculator input", "copay_percent" in estimate_claim.args_schema.model_json_schema()["required"])
check("chat history was sent with question 2", "Is cataract covered?" in sent_to_llm[2])
check("a new question is searched on its own", "My bill is 1.2 lakh, 8000/day room, 4 days. How much will I get?" in search_texts)
check("follow-up search also uses the previous question", search_texts[-1].startswith("Is cataract covered?"))
check("labels restart at P1 for each question", result2["sources"][0]["label"] == "P1")

everything = "".join(sent_to_llm)
check("phone from the PDF never reached the LLM", "9876543210" not in everything)
check("PAN from the PDF never reached the LLM", "ABCDE1234F" not in everything)
check("email from the PDF never reached the LLM", "ravi@example.com" not in everything)
check("Aadhaar typed by the user never reached the LLM", "2345 6789 0123" not in everything)

# ---------- 4. general question, no policy uploaded, answered from IRDAI rules ----------
rules_store = fake_store([Document(
    page_content="Co-payment means the policyholder bears a specified percentage of the claim.",
    metadata={"file": "health_master_circular.pdf", "page": 12, "clause": "Definitions"},
)])
llm = FakeLLM(replies=[
    AIMessage(content="", tool_calls=[{"name": "search_irdai_rules", "id": "3", "args": {"query": "co-payment"}}]),
    AIMessage(content="A co-payment is the share of the claim you pay yourself 【R1】."),
])
result3 = GraphBuilder(None, rules_store, llm).run("What is a co-payment?")
print("\nAnswer 3:", result3["answer"])
check("works with no policy uploaded", result3["sources"][0]["label"] == "R1")
check("no calculation box for a search-only answer", result3["calculations"] == [])
check("【R1】 brackets turned into [R1]", "[R1]" in result3["answer"] and "【" not in result3["answer"])
check("LLM was told no policy is uploaded", "has not uploaded a policy" in sent_to_llm[-2])

# ---------- 5. web search when IRDAI rules don't have it ----------
class FakeTavily:
    def __init__(self, results):
        self.results = results
        self.called = False

    def invoke(self, arguments):
        self.called = True
        return {"results": self.results}

llm = FakeLLM(replies=[
    AIMessage(content="", tool_calls=[{"name": "search_web", "id": "4", "args": {"query": "how to port policy"}}]),
    AIMessage(content="You can port your policy at renewal [W1]."),
])
web_graph = GraphBuilder(None, None, llm)
web_graph.nodes.trusted_web_search = FakeTavily([])  # official sites: nothing found
web_graph.nodes.open_web_search = FakeTavily([{"url": "https://example.com/port", "title": "Porting",
                                               "content": "Port at renewal. Helpline 9123456789."}])
result4 = web_graph.run("How do I port my policy?")
print("\nAnswer 4:", result4["answer"])
check("tried official websites first, then the whole web",
      web_graph.nodes.trusted_web_search.called and web_graph.nodes.open_web_search.called)
check("web result labelled W1 with its link", result4["sources"][0]["file"] == "https://example.com/port")
check("phone number in a web result was hidden", "9123456789" not in sent_to_llm[-1])

print("\nALL PASSED" if not failed else f"\n{len(failed)} FAILED")
sys.exit(1 if failed else 0)
