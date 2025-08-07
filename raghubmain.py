import os
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone




from configloader import load_env

# Load environment
config = load_env()
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# Connect to Pinecone and get index
pc = Pinecone(api_key=config["PINECONE_API_KEY"])
index = pc.Index(config["PINECONE_INDEX"])

# File loader function
def load_documents(file_paths):
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            elif ext in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(path)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(path, encoding="utf-8")
            else:
                print(f"❌ Unsupported file: {path}")
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(path)
            docs.extend(loaded_docs)

        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
    return docs

# Example usage
file_inputs = [
    "PDFs/Paper.pdf",
    "PDFs/Large_language_models.txt"
]

documents = load_documents(file_inputs)
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
chunks = splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large",  # ✅ 1024 dim, publicly available
    model_kwargs={"device": "cpu"}
)


sample_text = chunks[0].page_content
vector = embeddings.embed_query(sample_text)
print("🔢 Embedding vector size:", len(vector))


# ✅ Store in Pinecone using updated SDK + LangChain integration
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=config["PINECONE_INDEX"]
)



# Start interactive QA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

while True:
    query = input("\n💬 Ask a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print("🤖 Answer:", answer)
