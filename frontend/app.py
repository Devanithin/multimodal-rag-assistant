import streamlit as st

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multimodal RAG Assistant")
st.write("Upload your documents and chat with them using AI.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Documents")
    st.write("Go to the Upload page to add your files.")
    if st.button("Go to Upload →"):
        st.switch_page("pages/1_upload.py")

with col2:
    st.subheader("💬 Chat with Documents")
    st.write("Go to the Chat page to ask questions.")
    if st.button("Go to Chat →"):
        st.switch_page("pages/2_chat.py")