import streamlit as st
from pathlib import Path
import subprocess, sys
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Framo Manuals Chatbot", layout="wide")
st.title("Framo Manuals — PDF Retrieval Chatbot")
st.write("Select a model and ask questions. The bot answers using uploaded Framo PDF manuals.")

MODEL_OPTIONS = ["Framo"]
selected = st.selectbox("Select machine model", MODEL_OPTIONS)

if selected:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello — ask me anything about the Framo manuals."}
        ]

    # Handle safe clearing logic
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False
    if "input" not in st.session_state:
        st.session_state.input = ""

    # Clear text field on next run if needed
    if st.session_state.clear_input:
        st.session_state["input"] = ""
        st.session_state.clear_input = False

    # Layout columns
    cols = st.columns([3, 1])
    with cols[0]:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'**You:** {msg["content"]}')
            else:
                st.markdown(f'**Bot:** {msg["content"]}')

    with cols[1]:
        user_input = st.text_area("Your question", height=140, key="input")

        if st.button("Send"):
            if user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner("Querying manuals..."):
                    try:
                        from query import answer_query
                        answer, sources, snippets = answer_query(user_input, model_name="ollama")
                        sources_md = ""
                        if sources:
                            sources_md = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer + sources_md}
                        )
                        with st.expander("Context snippets used (click to expand)"):
                            for s in snippets:
                                st.write("- " + s)
                    except Exception as e:
                        st.error("Error while answering the query: " + str(e))
                        st.stop()

                # ✅ Safe clear after message send
                st.session_state.clear_input = True
                st.rerun()

    # Admin tools
    st.markdown("---")
    st.markdown("**Admin / System**")
    run_ingest = st.button("Re-run ingest (process all PDFs in data/)")
    if run_ingest:
        st.info("Running ingest.py — this may take a while depending on model/downloads.")
        subprocess.run([sys.executable, "ingest.py"], check=False)
        st.success("Ingest finished (check logs).")

    st.write("Place all Framo PDF manuals into the `data/` folder and re-run ingest.")