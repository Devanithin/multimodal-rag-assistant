import streamlit as st
import requests

st.set_page_config(page_title="Chat", page_icon="💬")

BACKEND_URL = "http://localhost:8000"

st.title("💬 Chat with your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption(f"📄 Sources: {', '.join(message['sources'])}")
        if message["role"] == "assistant" and message.get("latency"):
            st.caption(f"⚡ Total: {message['latency']['total_seconds']}s | Retrieval: {message['latency']['retrieval_seconds']}s | LLM: {message['latency']['llm_seconds']}s")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/query/ask",
                json={"question": question}
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                latency = data.get("latency", {})

                st.write(answer)
                if sources:
                    st.caption(f"📄 Sources: {', '.join(sources)}")
                if latency:
                    st.caption(f"⚡ Total: {latency['total_seconds']}s | Retrieval: {latency['retrieval_seconds']}s | LLM: {latency['llm_seconds']}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency": latency
                })
            else:
                st.error("❌ Something went wrong. Please try again.")

if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()