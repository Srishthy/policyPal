"""Build the IRDAI rules index. Run this once after putting the IRDAI PDFs in data/irdai/:

    python -m scripts.ingest_rules
"""

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vector_store import VectorStore


def main():
    doc_processor = DocumentProcessor(Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)

    documents = doc_processor.load_from_pdf_dir(Config.IRDAI_PDF_FOLDER)
    if not documents:
        print(f"No readable PDFs found in {Config.IRDAI_PDF_FOLDER}")
        return

    chunks = doc_processor.split_documents(documents)
    print(f"Made {len(chunks)} chunks. Creating embeddings (this can take a few minutes)...")

    vector_store = VectorStore()
    vector_store.create_vectorstore(chunks)
    vector_store.save(Config.IRDAI_INDEX_FOLDER)
    print(f"Saved the IRDAI index to {Config.IRDAI_INDEX_FOLDER}")


if __name__ == "__main__":
    main()
