import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.memory import ConversationBufferWindowMemory

CHROMA_DIR = "chroma_db"

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.memory = ConversationBufferWindowMemory(
            k=5,
            return_messages=True
        )

        # Load existing DB if exists
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings
        )

    def process_pdfs(self, file_paths):
        documents = []

        for path in file_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()

            # Add source metadata (important for multi-file)
            for d in docs:
                d.metadata["source"] = os.path.basename(path)

            documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(documents)

        # Add to Chroma (persistent automatically)
        self.vectorstore.add_documents(split_docs)

        return len(split_docs)

    def ask(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        history = "\n".join([m.content for m in self.memory.chat_memory.messages])

        prompt = f"""
You are a strict document assistant.

RULES:
- Answer ONLY from the given context
- If not found, say: "I don't know"
- Do NOT guess or hallucinate

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        self.memory.save_context(
            {"input": query},
            {"output": response.content}
        )

        sources = list(set([
            f"{d.metadata.get('source')} (p.{d.metadata.get('page',0)+1})"
            for d in docs
        ]))

        return {
            "answer": response.content,
            "sources": sources
        }