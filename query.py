"""
query.py - Retrieval and answer generation.
- Loads persisted Chroma DB (./db/chroma)
- Performs similarity search, collects top-k chunks with metadata (source + page)
- Calls OpenAI Chat (GPT) if API key present; otherwise raises instructive error.
- Returns (answer_text, sources_list, snippets)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent
DB_DIR = BASE / "db" / "chroma"
TOP_K = int(os.getenv("TOP_K", "4"))


def _load_embeddings():
    """Load embedding model (HuggingFace by default, OpenAI if configured)."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
    except Exception as e:
        raise RuntimeError("❌ LangChain embeddings not installed or missing dependencies.") from e

    use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    return HuggingFaceEmbeddings(
        model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )


def _get_llm():
    """Use only the local Ollama LLM for responses."""
    try:
        from langchain_community.llms import Ollama
        model = os.getenv("OLLAMA_MODEL", "llama3")
        print(f"🦙 Using local Ollama model: {model}")
        return Ollama(model=model)
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to initialize Ollama LLM. Ensure Ollama is running with `ollama serve`. Details: {e}"
        )


def answer_query(question: str, model_name: str = "openai"):
    """Retrieve most relevant text chunks and generate an answer."""
    if not Path(DB_DIR).exists():
        raise FileNotFoundError(
            f"Vector DB not found. Run ingest.py to create it. Expected at: {DB_DIR}"
        )

    try:
        from langchain_openai import ChatOpenAI, OpenAI
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except Exception as e:
        raise RuntimeError(f"LangChain import failed: {e}")

    # Load database and retrieve context
    embeddings = _load_embeddings()
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    docs = vectordb.similarity_search_with_score(question, k=TOP_K)

    contexts, sources, snippets = [], [], []
    for doc, score in docs:
        text = doc.page_content.strip()
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        citation = f"{src} (page {page})" if page else src
        sources.append(citation)

        snippet = text[:400].replace("\n", " ").strip()
        snippets.append(f"{citation}: {snippet}")
        contexts.append(text)

    unique_sources = list(dict.fromkeys(sources))

    # Define concise context-aware prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert assistant answering technical questions using ONLY the provided CONTEXT.\n"
            "Be concise (2–4 sentences). If the answer is not present in the context, say 'I don't know — check the manuals.'\n"
            "Do not make up information.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER:"
        ),
    )

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    combined_context = "\n\n---\n\n".join(contexts)
    answer = chain.invoke({"context": combined_context, "question": question})

    return answer.strip(), unique_sources, snippets