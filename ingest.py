"""
ingest.py - Ingest all PDFs from data/ into a Chroma vectorstore.
- Splits documents into chunks with page-level metadata (source filename + page number).
- Uses HuggingFace sentence-transformers by default, or OpenAI embeddings if configured.
- Persists vectorstore to ./db/chroma
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def ingest():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
    except Exception as e:
        raise RuntimeError("Required langchain components are missing. Install requirements.txt and retry.") from e

    BASE = Path(__file__).parent
    DATA_DIR = BASE / "data"
    DB_DIR = BASE / "db" / "chroma"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted([p for p in DATA_DIR.glob("*.pdf")])
    if not pdfs:
        print("No PDFs found in data/. Please add your Framo manuals and retry.")
        return

    docs = []
    for p in pdfs:
        print(f"Loading PDF: {p.name}")
        loader = PyPDFLoader(str(p))
        # load and keep page information
        pages = loader.load_and_split()  # uses page-level splitting
        # attach filename + page metadata
        for i, page in enumerate(pages, start=1):
            page.metadata["source"] = p.name
            if "page" not in page.metadata:
                page.metadata["page"] = i
        docs.extend(pages)

    print(f"Loaded {len(docs)} pages from {len(pdfs)} PDF(s). Now chunking for retrieval...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks. Building embeddings...")

    # choose embeddings
    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    if use_openai and os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI embeddings (via OpenAI API).")
        embeddings = OpenAIEmbeddings()
    else:
        model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"Using HuggingFace embeddings: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))
    vectordb.persist()
    print("✅ Ingestion complete. Vector DB saved to:", DB_DIR)


if __name__ == '__main__':
    ingest()