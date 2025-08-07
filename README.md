# 📄 Multi-Doc-QA-RAGHub

A lightweight RAG (Retrieval-Augmented Generation) application that allows users to **upload multiple documents** (PDF, TXT, DOCX, MD) and ask **natural language questions**. Powered by **LangChain**, **OpenAI**, **Pinecone**, and **Streamlit**.

---

## 🧠 Features

- 📎 Upload multiple documents at once
- 🔍 Ask questions and get AI-generated answers
- ⚡ Powered by LLMs + vector search
- 💾 Pinecone vector store integration
- 🔓 Open-source and easy to customize

---

## 🧰 Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)
- [Streamlit](https://streamlit.io/)

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ravi2505-de/Multi-Doc-QA-RAGHub.git
cd Multi-Doc-QA-RAGHub
2. Install Dependencies
pip install -r requirements.txt
3. Set Environment Variables
Create a file named .env in the root folder:
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=your-index-name
Note: .env should be excluded from Git using .gitignore.

4. Run the App

streamlit run app.py
📂 Supported File Types
.pdf

.txt

.docx

.md

💡 Use Case
This tool is helpful for:

Students analyzing research papers

Developers reading documentation

Teams uploading meeting notes or specs

Researchers comparing multiple sources

✍️ Author
Built by ravi mutthina
Connect on LinkedIn
