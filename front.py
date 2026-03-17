import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Chroma RAG Chatbot", layout="wide")
st.title("📄 Multi-PDF Chatbot (ChromaDB)")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Upload
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Process PDFs") and files:
    with st.spinner("Uploading..."):
        res = requests.post(
            f"{API_URL}/upload/",
            files=[("files", (f.name, f, "application/pdf")) for f in files]
        )
        st.success(res.json()["message"])

# Chat
query = st.chat_input("Ask your question...")

if query:
    st.session_state.chat.append(("user", query))

    res = requests.get(
        f"{API_URL}/ask/",
        params={"question": query}
    ).json()

    answer = res["answer"]
    sources = res["sources"]

    st.session_state.chat.append(("assistant", answer))

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"📄 Sources: {sources}")

# Display history
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)