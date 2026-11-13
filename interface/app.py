"""
app.py  —  Streamlit interface for the Civil Code RAG.

Launch:
  streamlit run interface/app.py
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_backend import LLMConfigError

# ── Config page ──────────────────────────────────────────────

st.set_page_config(
    page_title = "Civil Code Assistant",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
.source-card {
    background: #f8f9fa;
    border-left: 4px solid #1f4e79;
    padding: 10px 15px;
    margin: 8px 0;
    border-radius: 4px;
    font-size: 0.9em;
}
.score-badge {
    background: #1f4e79;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8em;
}
.disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    padding: 10px;
    border-radius: 6px;
    font-size: 0.85em;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── RAG initialization (cached) ─────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading the RAG model...")
def load_rag():
    from rag.rag_pipeline import CodeCivilRAG
    try:
        return CodeCivilRAG()
    except LLMConfigError as exc:
        st.error(f"❌ Invalid LLM configuration: {exc}")
        st.stop()


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.image("https://www.legifrance.gouv.fr/contenu/logo", width=120)
    st.title("⚖️ Civil Code RAG")
    st.caption("Qwen3.8 × Qdrant × Légifrance")

    st.divider()

    top_k = st.slider("Retrieved articles (Top K)", 2, 10, 5)
    show_sources = st.toggle("Show sources", value=True)

    st.divider()
    st.subheader("💡 Example questions")
    examples = [
        "What are the conditions for marriage?",
        "What is tortious civil liability?",
        "How does extinctive prescription work?",
        "What are a tenant's rights?",
        "What is a sales contract?",
        "Explain the concept of force majeure",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question_input"] = ex

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Header ───────────────────────────────────────────────────

st.title("⚖️ French Civil Code Assistant")
st.caption("Ask your questions about French civil law. Answers are based on the official texts from Légifrance.")

# ── History ──────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} articles consulted"):
                for s in msg["sources"]:
                    st.markdown(f"""<div class="source-card">
                        <strong>Article {s['num']}</strong>
                        <span class="score-badge">{s['score']}</span>
                        {'<br><em>' + s['titre'] + '</em>' if s['titre'] else ''}
                        <br><small>{s['breadcrumb']}</small>
                        <br><br>{s['extrait']}
                    </div>""", unsafe_allow_html=True)


# ── Input ────────────────────────────────────────────────────

default_q = st.session_state.pop("question_input", "")
question  = st.chat_input("Ask your question about the Civil Code...",)

# Allow questions via the sidebar buttons
if not question and default_q:
    question = default_q


# ── Processing ───────────────────────────────────────────────

if question:
    # Display the question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("⏳ Consulting the Civil Code..."):
            try:
                rag = load_rag()
                st.caption(f"🤖 Backend: {rag.backend.mode} — {rag.model}")

                # Pass the history
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1][-6:]
                ]

                result = rag.ask(question, history=history)

                st.markdown(result["answer"])

                if show_sources and result["sources"]:
                    with st.expander(f"📚 {len(result['sources'])} articles consulted"):
                        for s in result["sources"]:
                            st.markdown(f"""<div class="source-card">
                                <strong>Article {s['num']}</strong>
                                <span class="score-badge">{s['score']}</span>
                                {'<br><em>' + s['titre'] + '</em>' if s['titre'] else ''}
                                <br><small>{s['breadcrumb']}</small>
                                <br><br>{s['extrait']}
                            </div>""", unsafe_allow_html=True)

                # Add to the history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as e:
                st.error(f"❌ Error: {e}\n\nCheck that Qdrant is running and that the LLM backend (local llama.cpp or external API) is configured.")

# ── Disclaimer ───────────────────────────────────────────────

st.markdown("""<div class="disclaimer">
⚠️ <strong>Legal disclaimer</strong>: This assistant is a tool to help understand civil law.
It does not replace the advice of a lawyer or a qualified legal professional.
For any personal legal question, consult a legal professional.
</div>""", unsafe_allow_html=True)
