import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
st.set_page_config(page_title="EduBot: Academic Research Assistant", page_icon="📚", layout="wide")

# ===== Sidebar =====
with st.sidebar:
    st.subheader("📑 Enter Research Paper URLs")
    urls = [st.text_input(f"🔗 Paper URL {i+1}") for i in range(3)]
    process_url_clicked = st.button("🚀 Process Papers from URLs")

    st.markdown("---")
    st.subheader("📂 Upload PDF or Text Files")
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "txt"], accept_multiple_files=True)
    process_file_clicked = st.button("🚀 Process Uploaded Files")

file_path = "edu_faiss_store.pkl"
progress_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.2, max_tokens=1500, model_name="gpt-3.5-turbo")

# ===== Vectorstore Building (URL) =====
if process_url_clicked:
    with st.spinner("Fetching and processing papers from URLs..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        progress_placeholder.progress(33, "Loaded Papers ✅")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=3000
        )
        docs = text_splitter.split_documents(data)
        progress_placeholder.progress(66, "Split Text ✅")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        progress_placeholder.progress(100, "Vectorstore Built ✅")

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Papers Processed Successfully from URLs!")

# ===== Vectorstore Building (File Upload) =====
if process_file_clicked:
    if not uploaded_files:
        st.error("❌ Please upload at least one file.")
    else:
        all_docs = []
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                if file_name.endswith(".txt"):
                    text = uploaded_file.read().decode("utf-8")
                    docs = [Document(page_content=text, metadata={"source": file_name})]
                elif file_name.endswith(".pdf"):
                    temp_path = f"temp_{file_name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    os.remove(temp_path)
                else:
                    continue
                all_docs.extend(docs)

            progress_placeholder.progress(33, "Loaded Files ✅")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=3000
            )
            docs = text_splitter.split_documents(all_docs)
            progress_placeholder.progress(66, "Split Text ✅")

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            progress_placeholder.progress(100, "Vectorstore Built ✅")

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

            st.success("✅ Files Processed Successfully!")

# ====== Chat Assistant GUI ======
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_response(user_query):
    if not os.path.exists(file_path):
        return "⚠️ Please process papers first (via URL or file upload)."

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), return_source_documents=True
    )
    result = chain.invoke({"query": user_query})
    answer = result['result']
    return answer

# Render chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if user_input := st.chat_input("💬 Ask your academic question..."):
    st.chat_message("user").markdown(user_input)
    response = get_response(user_input)
    st.chat_message("assistant").markdown(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})


