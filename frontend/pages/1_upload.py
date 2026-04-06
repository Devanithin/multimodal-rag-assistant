import streamlit as st
import requests

st.set_page_config(page_title="Upload Files", page_icon="📄")

BACKEND_URL = "http://localhost:8000"

st.title("📄 Upload Content")
st.caption("Add documents, media files, YouTube videos or web pages to your knowledge base.")

st.markdown("---")

# ── Tabs ──
tab1, tab2 = st.tabs(["📁 File Upload", "🌐 URL / YouTube"])

# ── Tab 1 — File Upload ──
with tab1:
    st.subheader("Upload a File")
    st.caption("Supported: PDF, CSV, XLSX, DOCX, MP3, MP4, WAV, MOV, M4A")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "csv", "xlsx", "docx", "mp3", "mp4", "wav", "mov", "m4a"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📄 {uploaded_file.name} — {round(uploaded_file.size / 1024, 1)} KB")
        with col2:
            if st.button("Upload", use_container_width=True):
                with st.spinner("Processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/files/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Done! {data['chunk_count']} chunks created")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {response.text}")

# ── Tab 2 — URL / YouTube ──
with tab2:
    st.subheader("Load from URL")
    st.caption("Paste a YouTube link or any webpage URL")

    url_input = st.text_input("URL", placeholder="https://youtube.com/watch?v=... or https://example.com")
    url_name = st.text_input("Name (optional)", placeholder="Give it a friendly name")

    if st.button("Load Content", use_container_width=True):
        if url_input:
            with st.spinner("Extracting content..."):
                response = requests.post(
                    f"{BACKEND_URL}/files/upload-url",
                    json={"url": url_input, "name": url_name or url_input}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Done! {data['chunk_count']} chunks created from {data['file_type'].upper()}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {response.text}")
        else:
            st.warning("⚠️ Please enter a URL")

st.markdown("---")

# ── Uploaded Files List ──
st.subheader("📚 Knowledge Base")

response = requests.get(f"{BACKEND_URL}/files/")
if response.status_code == 200:
    files = response.json()
    if not files:
        st.info("No content added yet. Upload a file or load a URL above.")
    else:
        st.caption(f"{len(files)} item(s) in your knowledge base")
        for f in files:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                with col1:
                    icons = {
                        "pdf": "📕", "csv": "📊", "xlsx": "📊",
                        "docx": "📝", "mp3": "🎵", "mp4": "🎬",
                        "wav": "🎵", "mov": "🎬", "m4a": "🎵",
                        "youtube": "▶️", "web": "🌐"
                    }
                    icon = icons.get(f['file_type'], "📄")
                    st.write(f"{icon} {f['filename']}")
                with col2:
                    st.caption(f['file_type'].upper())
                with col3:
                    st.caption(f"{round(f['size'] / 1024, 1)} KB")
                with col4:
                    st.caption(f"🧩 {f['chunk_count']}")
                with col5:
                    if st.button("🗑️", key=f"del_{f['id']}", help="Delete"):
                        del_response = requests.delete(f"{BACKEND_URL}/files/{f['id']}")
                        if del_response.status_code == 200:
                            st.rerun()