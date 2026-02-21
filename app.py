import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Advanced AI RAG", layout="centered")
st.title("🚀 Advanced Document RAG (DeepLearning.AI Edition)")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document with ChromaDB..."):
        process_document_to_chroma_db(uploaded_file.name)
    st.success("Document Indexed Successfully!")

# User Input
user_question = st.text_area("Ask a complex question about the document")

if st.button("Generate Answer"):
    with st.spinner("Expanding Query & Re-ranking Results..."):
        answer = answer_question(user_question)

    st.markdown("### 🤖 Final Expert Response")
    st.write(answer)

    st.info("Technical Note: This answer was refined using Query Expansion and Cross-Encoder Re-ranking.")