from fastapi import FastAPI, UploadFile, File
import shutil
import os
from rag import RAGEngine

app = FastAPI()
rag = RAGEngine()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload(files: list[UploadFile] = File(...)):
    paths = []

    for file in files:
        path = f"{UPLOAD_DIR}/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(path)

    count = rag.process_pdfs(paths)

    return {"message": f"{count} chunks added to ChromaDB."}


@app.get("/ask/")
def ask(question: str):
    return rag.ask(question)