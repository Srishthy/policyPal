"""Streamlit UI for PolicyPal (built on the course's Agentic RAG app)"""

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vector_store import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🩺 PolicyPal",
    page_icon="🩺",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'policy_name' not in st.session_state:
        st.session_state.policy_name = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'uploaded_key' not in st.session_state:
        st.session_state.uploaded_key = []   # which files were read last time


@st.cache_resource
def initialize_rag():
    """Load the embedding model and the IRDAI rules once (cached)"""
    rules_store = VectorStore()  # this also loads the embedding model
    if rules_store.load(Config.IRDAI_INDEX_FOLDER):
        return rules_store.embedding, rules_store
    return rules_store.embedding, None


def process_policy(uploaded_files):
    """Read the uploaded policy PDFs and build a RAG system for them"""
    embedding, rules_store = initialize_rag()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    # the PDF loader needs files on disk, so save the uploads in a temporary folder
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for uploaded_file in uploaded_files:
            path = Path(folder) / uploaded_file.name
            path.write_bytes(uploaded_file.getvalue())
            paths.append(path)
        documents = doc_processor.process_pdfs(paths)

    policy_store = VectorStore(embedding=embedding)
    policy_store.create_vectorstore(documents)

    graph_builder = GraphBuilder(policy_store, rules_store, Config.get_llm())
    graph_builder.build()
    return graph_builder, len(documents)


def general_rag_system():
    """Before a policy is uploaded: answers general questions (IRDAI rules + web)"""
    embedding, rules_store = initialize_rag()
    graph_builder = GraphBuilder(None, rules_store, Config.get_llm())
    graph_builder.build()
    return graph_builder


def show_sources(answer, sources):
    """Show the clauses / web pages the answer cites (or everything searched if it cites none)"""
    cited = [s for s in sources if f"[{s['label']}]" in answer]
    to_show = cited if cited else sources
    if not to_show:
        return

    with st.expander(f"📄 Sources ({len(to_show)})"):
        for source in to_show:
            label = source["label"]
            if label.startswith("W"):
                st.markdown(f"**[{label}] Web** · [{source['clause'] or source['file']}]({source['file']})")
            else:
                where = "Your policy" if label.startswith("P") else "IRDAI rules"
                st.markdown(f"**[{label}] {where}** · {source['file']} · page {source['page']} · {source['clause']}")
            st.caption(source["text"][:500] + "...")


def sidebar():
    with st.sidebar:
        st.header("Your policy")
        uploaded_files = st.file_uploader("Upload your policy PDF", type="pdf", accept_multiple_files=True)

        # read the policy automatically when new files are uploaded
        uploaded_key = [(f.name, f.size) for f in uploaded_files]
        if uploaded_files and uploaded_key != st.session_state.uploaded_key:
            st.session_state.uploaded_key = uploaded_key
            with st.spinner("Reading your policy..."):
                try:
                    rag_system, num_chunks = process_policy(uploaded_files)
                    st.session_state.rag_system = rag_system
                    st.session_state.policy_name = ", ".join(f.name for f in uploaded_files)
                    st.session_state.history = []
                    st.success(f"✅ Policy ready! ({num_chunks} chunks)")
                except Exception as e:
                    st.error(f"Failed to read the policy: {str(e)}")

        # the user removed the files: go back to general questions only
        if not uploaded_files and st.session_state.policy_name:
            st.session_state.rag_system = None
            st.session_state.policy_name = None
            st.session_state.uploaded_key = []
            st.session_state.history = []

        if st.button("New chat"):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
        embedding, rules_store = initialize_rag()
        st.caption("✅ IRDAI rules loaded" if rules_store else
                   "IRDAI rules not loaded (run `python -m scripts.ingest_rules`)")
        st.caption("✅ Web search on" if Config.TAVILY_API_KEY else
                   "Web search off (add TAVILY_API_KEY to .env)")
        st.caption("Aadhaar, PAN, phone and email are hidden before anything is sent to the AI.")


def main():
    """Main application"""
    init_session_state()

    if not Config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Add it to the .env file.")
        return

    # Title
    st.title("🩺 PolicyPal")
    st.markdown("Ask about your health insurance policy, or any general health insurance question")

    sidebar()

    # Initialize system (general questions only, until a policy is uploaded)
    if st.session_state.rag_system is None:
        with st.spinner("Loading system..."):
            st.session_state.rag_system = general_rag_system()

    if st.session_state.policy_name:
        st.caption(f"📄 Policy loaded: {st.session_state.policy_name}")
    else:
        st.caption("No policy uploaded yet. You can still ask general questions.")

    st.markdown("---")

    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. Is cataract surgery covered? What is a co-payment?"
        )
        submit = st.form_submit_button("🔍 Ask")

    # Process question
    if submit and question:
        with st.spinner("Searching..."):
            start_time = time.time()

            try:
                result = st.session_state.rag_system.run(question, st.session_state.history)
            except Exception as e:
                message = str(e)
                if "429" in message or "rate" in message.lower():
                    message = "Groq's free limit was reached. Wait a minute and try again."
                st.error(f"Something went wrong: {message}")
                return

            elapsed_time = time.time() - start_time

            # Add to history
            st.session_state.history.append({
                'question': question,
                'answer': result['answer'],
                'time': elapsed_time
            })

            # Display answer
            st.markdown("### 💡 Answer")
            st.success(result['answer'])

            show_sources(result['answer'], result['sources'])

            if result['calculations']:
                with st.expander("🧮 How we calculated this"):
                    for calculation in result['calculations']:
                        st.write(calculation)

            st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Questions")

        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")


if __name__ == "__main__":
    main()
