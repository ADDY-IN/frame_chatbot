# Framo Chatbot — Multi-PDF Support

This updated prototype supports multiple Framo manuals (PDFs). Place all your manuals into `data/` (e.g., `09- Cargo Heating- Operational Advice.pdf`, etc.) and run `python ingest.py` to create the Chroma vector DB.

## How it works
- Ingest: `ingest.py` loads all PDFs, records filename and page number for each page, chunks text, embeds and persists to `./db/chroma`.
- Query: `query.py` performs similarity search, returns top-k relevant chunks, and calls the configured LLM to generate a concise answer. It also returns short snippets and file+page citations for transparency.
- App: `app.py` is a Streamlit UI showing a dropdown (Framo) + chat. Answers include a source list and an expander with context snippets.

## Quickstart
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Put all your PDF manuals in `data/`.
3. (Optional) configure `.env` (OPENAI_API_KEY, USE_OPENAI_EMBEDDINGS=true, OPENAI_MODEL=gpt-4)
4. Ingest:
```bash
python ingest.py
```
5. Run the app:
```bash
streamlit run app.py
```

## Notes
- The project defaults to HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- If using OpenAI for embeddings/LLM, ensure `OPENAI_API_KEY` is set in `.env`.
- For local LLMs (Ollama), adapt `_get_llm()` in `query.py` accordingly.
