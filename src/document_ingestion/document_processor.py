from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.config import Config


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a PDF file, one Document per page"""
        file_path = Path(file_path)
        pages = PyMuPDFLoader(str(file_path)).load()

        # keeping  only the metadata we need: file name and page number (starting from 1)
        for page in pages:
            page.metadata = {"file": file_path.name, "page": page.metadata["page"] + 1}

        # a scanned PDF has (almost) no text, so we can't search it
        empty_pages = [page for page in pages if len(page.page_content.strip()) < 50]
        if len(empty_pages) > len(pages) / 2:
            raise ValueError(
                f"{file_path.name} looks like a scanned PDF with no readable text. "
                "Please download the digital PDF from your insurer's website or app."
            )
        return pages

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        docs: List[Document] = []
        for pdf_path in sorted(Path(directory).glob("*.pdf")):
            try:
                pages = self.load_from_pdf(pdf_path)
                docs.extend(pages)
                print(f"Loaded {pdf_path.name}: {len(pages)} pages")
            except ValueError as error:
                print(f"Skipped {pdf_path.name}: {error}")
        return docs

    def is_clause_heading(self, line: str) -> bool:
        """Guess if a line is a clause heading """
        line = line.strip()
        if len(line) < 4 or len(line) > 80:
            return False

        first_word = line.split()[0]
        rest_of_line = line[len(first_word):].strip()

        # page footers like "26 | Page" or "25 I P age" (scanned text) are not headings
        if "page" in line.lower().replace(" ", ""):
            return False

        # "4.2 Room Rent" or "5. Waiting Periods"
        if first_word.replace(".", "").isdigit() and rest_of_line[:1].isupper():
            return True

        # "Section 3 - Benefits" or "Clause 7"
        if first_word.lower() in ["section", "clause", "part", "chapter"]:
            return True

        # "Excl01 Cosmetic surgery..." (standard exclusion codes)
        if first_word.lower().startswith("excl") and any(char.isdigit() for char in first_word):
            return True

        return False

    def add_clause_headings(self, chunks: List[Document]) -> List[Document]:
        """
        Tag every chunk with the clause it belongs to, so answers can say
        "page 2, 4.2 Room Rent". A clause continues until the next heading.
        """
        current_clause = ""
        current_file = ""

        for chunk in chunks:
            # a new file starts without a clause
            if chunk.metadata["file"] != current_file:
                current_file = chunk.metadata["file"]
                current_clause = ""

            lines = chunk.page_content.strip().splitlines()

            # a chunk that starts with a heading belongs to that clause
            if lines and self.is_clause_heading(lines[0]):
                current_clause = lines[0].strip()
            chunk.metadata["clause"] = current_clause

            # the last heading inside this chunk is where the next chunk continues
            for line in lines:
                if self.is_clause_heading(line):
                    current_clause = line.strip()

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        chunks = self.splitter.split_documents(documents)
        return self.add_clause_headings(chunks)

    def process_pdfs(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Complete pipeline to load and split PDF files

        Args:
            file_paths: List of PDF paths to process

        Returns:
            List of processed document chunks
        """
        docs: List[Document] = []
        for file_path in file_paths:
            docs.extend(self.load_from_pdf(file_path))
        return self.split_documents(docs)
