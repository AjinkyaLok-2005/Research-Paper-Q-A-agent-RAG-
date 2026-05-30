import os
import tempfile

import streamlit as st

from config.settings import UPLOADED_PDFS_DIR
from core.ingestion  import ingest_multiple_pdfs
from core.indexing   import embed_and_index, get_indexed_papers, get_collection_stats
from core.agent      import decompose_question
from core.retrieval  import retrieve_for_all_subquestions
from core.synthesis  import synthesize_answer


st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide",
)



if "indexed_papers" not in st.session_state:
    st.session_state.indexed_papers = get_indexed_papers()

if "last_result" not in st.session_state:
    st.session_state.last_result = None



with st.sidebar:
    st.title("📄 Research Paper Q&A")
    st.caption("Powered by Groq · ChromaDB · LangGraph")
    st.divider()

    st.subheader("Upload Research Papers")

    uploaded_files = st.file_uploader(
        label="Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs only. Scanned image PDFs are not supported.",
    )

    if uploaded_files:
        if st.button("Index Uploaded PDFs", type="primary", use_container_width=True):
            os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)

            with st.spinner("Ingesting and indexing PDFs ..."):
                # Save uploaded files to disk temporarily
                saved_paths = []
                for uf in uploaded_files:
                    dest = os.path.join(UPLOADED_PDFS_DIR, uf.name)
                    with open(dest, "wb") as f:
                        f.write(uf.read())
                    saved_paths.append(dest)

                # Ingest → chunk → embed → index
                try:
                    chunks = ingest_multiple_pdfs(saved_paths)
                    if chunks:
                        embed_and_index(chunks)
                        st.session_state.indexed_papers = get_indexed_papers()
                        st.success(
                            f"Indexed {len(chunks)} chunks from "
                            f"{len(uploaded_files)} PDF(s)."
                        )
                    else:
                        st.error(
                            "No text could be extracted from the uploaded PDFs. "
                            "Please check that they are not scanned image PDFs."
                        )
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()

    st.subheader("Indexed Papers")

    if st.session_state.indexed_papers:
        for paper in st.session_state.indexed_papers:
            st.markdown(f"✅ `{paper}`")

        stats = get_collection_stats()
        st.caption(
            f"{stats.get('total_chunks', 0)} total chunks across "
            f"{stats.get('total_papers', 0)} paper(s)"
        )
    else:
        st.info("No papers indexed yet. Upload PDFs above.")

    if st.button("Refresh Paper List", use_container_width=True):
        st.session_state.indexed_papers = get_indexed_papers()
        st.rerun()



st.title("Research Paper Q&A Agent")
st.markdown(
    "Ask any question about your uploaded research papers. "
    "Complex questions are automatically broken into sub-questions for deeper retrieval."
)
st.divider()


question = st.text_area(
    label="Your Question",
    placeholder=(
        "e.g. What are the main contributions of the paper?\n"
        "e.g. How does the proposed method compare to baselines and what datasets were used?"
    ),
    height=100,
    key="question_input",
)

ask_col, clear_col = st.columns([3, 1])

with ask_col:
    ask_clicked = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

with clear_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()


if ask_clicked and question.strip():

    if not st.session_state.indexed_papers:
        st.warning("Please upload and index at least one PDF before asking questions.")
    else:
        with st.spinner("Running query decomposition agent ..."):
            sub_questions = decompose_question(question)

        with st.spinner("Retrieving relevant chunks ..."):
            retrieval_results = retrieve_for_all_subquestions(sub_questions)

        with st.spinner("Synthesizing final answer ..."):
            result = synthesize_answer(question, retrieval_results)

        st.session_state.last_result = {
            "question":      question,
            "sub_questions": sub_questions,
            "retrieval":     retrieval_results,
            "answer":        result["answer"],
            "sources":       result["sources"],
        }


if st.session_state.last_result:
    res = st.session_state.last_result

    st.subheader("🔍 Query Decomposition")

    if len(res["sub_questions"]) == 1:
        st.info("This was identified as a **simple question** — no decomposition needed.")
        st.markdown(f"**Sub-question:** {res['sub_questions'][0]}")
    else:
        st.info(
            f"This was identified as a **complex question** — "
            f"decomposed into {len(res['sub_questions'])} sub-questions."
        )
        for i, sq in enumerate(res["sub_questions"], 1):
            st.markdown(f"**{i}.** {sq}")

    st.divider()

    st.subheader("💡 Answer")
    st.markdown(res["answer"])

    st.divider()

    st.subheader("📚 Sources Cited")

    if res["sources"]:
        for source in res["sources"]:
            st.markdown(
                f"- **{source['paper_name']}** — Page {source['page_number']}"
            )
    else:
        st.markdown("_No sources could be extracted from the answer._")

    st.divider()

    with st.expander("🔎 View Retrieved Chunks (Debug)", expanded=False):
        for i, r in enumerate(res["retrieval"], 1):
            st.markdown(f"**Sub-question {i}:** {r['sub_question']}")
            if r["chunks"]:
                for chunk in r["chunks"]:
                    st.markdown(
                        f"📄 `{chunk['paper_name']}` — "
                        f"Page {chunk['page_number']} "
                        f"_(distance: {chunk['distance']})_"
                    )
                    st.caption(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                    st.markdown("---")
            else:
                st.warning(f"No chunks retrieved for sub-question {i}.")