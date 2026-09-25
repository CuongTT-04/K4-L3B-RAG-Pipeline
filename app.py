"""Streamlit chat UI for the EU AI Act RAG assistant."""

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()
st.set_page_config(page_title="EU AI Act Assistant", page_icon="⚖️", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn đã dùng ({len(sources)})", expanded=True):
        for index, source in enumerate(sources, 1):
            metadata = source["metadata"]
            st.markdown(f"**S{index} · {metadata['title']}**")
            cols = st.columns([2, 1, 1])
            cols[0].caption(metadata["source"])
            cols[1].caption(f"method: {source['retrieval_method']}")
            cols[2].caption(f"score: {source['score']:.4f}")
            if metadata.get("url"):
                st.markdown(f"[Mở nguồn gốc]({metadata['url']})")
            st.caption(source["content"][:320].replace("\n", " ") + "…")


with st.sidebar:
    st.title("⚖️ EU AI Act Assistant")
    st.caption("Trợ lý hỏi đáp có dẫn nguồn về EU AI Act và nghĩa vụ đối với mô hình GPAI.")
    top_k = st.slider("Số chunks", 3, 10, 5)
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.info("Câu trả lời chỉ dựa trên corpus. Hệ thống sẽ từ chối khi không đủ bằng chứng.")

st.title("EU AI Act RAG Chatbot")
st.caption("Hybrid retrieval (dense + BM25 + RRF), vectorless fallback và citation truy vết được.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

if query := st.chat_input("Ví dụ: GPAI providers có nghĩa vụ minh bạch nào?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và kiểm chứng nguồn..."):
            result = generate_with_citation(query, top_k=top_k)
        st.markdown(result["answer"])
        st.caption(f"retrieval source: {result['retrieval_source']}")
        show_sources(result["sources"])
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "retrieval_source": result["retrieval_source"],
    })
