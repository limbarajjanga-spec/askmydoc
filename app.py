# app.py
import streamlit as st
import tempfile
import os

from rag.ingestion import load_and_chunk_file
from rag.embeddings import embed_chunks, embed_query
from rag.vectorstore import store_chunks, clear_collection, get_collection_count
from rag.retriever import retrieve_similar_chunks
from llm.claude_client import ask_claude

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Document Assistant")
st.caption("Upload documents and ask questions — powered by Claude")

# ── Session state ─────────────────────────────────────────
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}   # {filename: chunk_count}

if "active_doc" not in st.session_state:
    st.session_state.active_doc = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # full conversation history

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    st.markdown("Supported: **PDF, TXT, DOCX**")

    uploaded_file = st.file_uploader(
        label="Upload a document",
        type=["pdf", "txt", "docx"],
        help="Upload PDF, TXT or DOCX files"
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.uploaded_docs:
            if st.button("Process File", type="primary", use_container_width=True):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=f".{uploaded_file.name.split('.')[-1]}"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        # Extract + chunk (auto-detects file type)
                        chunks, metadatas = load_and_chunk_file(
                            tmp_path, uploaded_file.name
                        )

                        # Embed
                        vectors = embed_chunks(chunks)

                        # Store — each doc gets its own chunks in ChromaDB
                        # We DON'T clear — we ADD to existing collection
                        store_chunks(
                            chunks, vectors,
                            uploaded_file.name, metadatas
                        )

                        os.unlink(tmp_path)

                        # Track uploaded docs
                        st.session_state.uploaded_docs[uploaded_file.name] = len(chunks)
                        st.session_state.active_doc = uploaded_file.name
                        st.session_state.chat_history = []

                        st.success(f"Indexed {len(chunks)} chunks!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info(f"Already indexed!")

    # ── Document selector ─────────────────────────────────
    if st.session_state.uploaded_docs:
        st.divider()
        st.markdown("**Indexed documents:**")

        for doc_name, chunk_count in st.session_state.uploaded_docs.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    f"📄 {doc_name[:25]}",
                    use_container_width=True,
                    type="primary" if doc_name == st.session_state.active_doc else "secondary"
                ):
                    st.session_state.active_doc = doc_name
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                st.caption(f"{chunk_count}c")

        st.divider()
        if st.button("Clear all docs", use_container_width=True):
            clear_collection()
            st.session_state.uploaded_docs = {}
            st.session_state.active_doc = None
            st.session_state.chat_history = []
            st.rerun()

# ── Main chat area ────────────────────────────────────────
if not st.session_state.uploaded_docs:
    st.info("Upload a document from the sidebar to get started.")
    with st.expander("How it works"):
        st.markdown("""
        1. **Upload** a PDF, TXT or DOCX from the sidebar
        2. **Wait** for it to be processed and indexed
        3. **Ask** any question about the document
        4. **Get** answers with page numbers and sources
        5. **Upload more** documents and switch between them
        """)

else:
    active = st.session_state.active_doc
    st.markdown(f"### Chatting about `{active}`")
    st.caption(f"{st.session_state.uploaded_docs.get(active, 0)} chunks indexed — "
               f"{len(st.session_state.chat_history)//2} questions asked")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input("Ask a question about your document...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        # Build history to send to Claude (exclude current question)
        history_for_claude = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        # Add to session history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    # Embed query
                    query_vector = embed_query(question)

                    # Retrieve chunks — filter by active doc
                    chunks = retrieve_similar_chunks(
                        query_vector,
                        source_filter=active
                    )

                    # Ask Claude with context + chat history
                    answer = ask_claude(question, chunks, history_for_claude)

                    st.markdown(answer)

                    # Sources panel
                    with st.expander("View sources"):
                        for i, chunk in enumerate(chunks):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.markdown(f"**Page {chunk['page']}**")
                                st.caption(f"Score: {chunk['score']}")
                            with col2:
                                st.caption(chunk["text"][:300] + "...")
                            st.divider()

                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })