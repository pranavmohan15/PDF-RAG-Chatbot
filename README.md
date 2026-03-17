# 📄 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to query information from PDF documents using LLMs.

---

## 🚀 Features

- Upload and process multiple PDF files  
- Semantic search using embeddings  
- Context-aware question answering  
- Source citation with page numbers  
- Conversation memory (last 5 interactions)  
- Persistent vector database using Chroma  

---

## 🧠 How It Works

1. PDFs are loaded and split into smaller chunks  
2. Each chunk is converted into embeddings  
3. Embeddings are stored in a vector database (Chroma)  
4. User queries are converted into embeddings  
5. Relevant chunks are retrieved using similarity search  
6. LLM generates answers strictly based on retrieved context  

---

## 🛠️ Tech Stack

- Python  
- LangChain  
- OpenAI (Embeddings + LLM)  
- ChromaDB (Vector Store)  

---

## 📂 Project Structure
