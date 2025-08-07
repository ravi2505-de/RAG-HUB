import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone
from configloader import load_env

# --- Page config ---
st.set_page_config(page_title="RAGHub", page_icon="📄", layout="wide")

# --- Sidebar Info ---
st.sidebar.title("📚 RAGHub")
st.sidebar.markdown("""
**👨‍💻 Built by Ravi Mutthina**

🔍 A GenAI-powered app to query PDFs, TXT, DOCX, and Markdown files using LLMs + Pinecone.

**Stack:** LangChain, Pinecone, OpenAI, HuggingFace, Streamlit

**Use case:** Ask questions from research papers, tech docs, or notes.
""")

# --- Load environment vars ---
config = load_env()
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
pc = Pinecone(api_key=config["PINECONE_API_KEY"])
index_name = config["PINECONE_INDEX"]

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",  # ✅ 1024 dim, publicly available
    model_kwargs={"device": "cpu"}
)

# --- Main Title ---
st.title("📄 RAGHub: Research Paper Q&A with GenAI")

# --- Upload Area ---
st.subheader("📎 Upload your documents")
uploaded_files = st.file_uploader(
    "Supported: .pdf, .txt, .md, .docx",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True
)

# --- Tabs ---
chat_tab, files_tab = st.tabs(["💬 Chat with Documents", "📁 Uploaded Files"])

with files_tab:
    if uploaded_files:
        st.markdown("### 🗂 Uploaded Files")
        for file in uploaded_files:
            st.markdown(f"- 📄 **{file.name}** ({round(file.size / 1024, 2)} KB)")
    else:
        st.info("Upload documents to see them listed here.")

with chat_tab:
    if uploaded_files:
        with st.spinner("🔍 Processing and indexing your files..."):
            documents = []
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[-1].lower()
                try:
                    if ext == ".pdf":
                        with open(file.name, "wb") as f:
                            f.write(file.getvalue())
                        loader = PyPDFLoader(file.name)
                    elif ext in [".txt", ".md"]:
                       # Read raw text from the uploaded file directly
                        raw_text = file.read().decode("utf-8")
                        from langchain.schema import Document
                        doc = Document(page_content=raw_text, metadata={"source": file.name})
                        documents.append(doc)
                        continue  # skip to next file
                    elif ext in [".docx", ".doc"]:
                        with open(file.name, "wb") as f:
                          f.write(file.getvalue())
                        loader = UnstructuredWordDocumentLoader(file.name)
                    else:
                        st.warning(f"Unsupported file: {file.name}")
                        continue

                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = file.name
                    documents.extend(docs)

                except Exception as e:
                    st.error(f"❌ Failed to load {file.name}: {e}")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=index_name
            )

            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            st.success("✅ Documents indexed. Ask your question below.")

            prompt = st.chat_input("Ask a question about your uploaded documents...")
            if prompt:
                with st.spinner("🤖 Thinking..."):
                    response = qa.run(prompt)
                    st.chat_message("user").markdown(prompt)
                    st.chat_message("ai").markdown(response)
    else:
        st.info("Upload files to enable the chat feature.")
