from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
from typing import List
import socket
import uvicorn

# LangChain and RAG Imports
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(title="RagResume API Backend")

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent
RESUMES_DIR = BASE_DIR / "Resumes"
INITIAL_RESUMES_DIR = BASE_DIR / "data" / "Resume" / "Resume_File (File responses)"
CHROMA_DIR = BASE_DIR / "chroma.db"
HOST = os.getenv("HOST", "127.0.0.1")


def get_chroma_dir() -> Path:
    candidates = [
        BASE_DIR / "chroma.db",
        Path.cwd() / "chroma.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_port(requested_port: int) -> int:
    for port in [requested_port, *range(requested_port + 1, requested_port + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found for the FastAPI backend.")


PORT = resolve_port(int(os.getenv("PORT", "8000")))

# Setup folder structure on startup
@app.on_event("startup")
def startup_event():
    # Create target Resumes folder
    RESUMES_DIR.mkdir(exist_ok=True)
    
    # Auto-copy initial resumes if the Resumes folder is empty and source exists
    if INITIAL_RESUMES_DIR.exists():
        existing_resumes = list(RESUMES_DIR.glob("*"))
        if len(existing_resumes) == 0:
            print("Populating initial resumes...")
            for file in INITIAL_RESUMES_DIR.glob("*"):
                if file.is_file():
                    shutil.copy(file, RESUMES_DIR / file.name)
            print(f"Copied {len(list(RESUMES_DIR.glob('*')))} resumes.")

# API Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    source_content: str = ""
    source_metadata: dict = {}

# Endpoints
@app.get("/resumes")
def get_resumes():
    """List all resumes uploaded in the Resumes directory."""
    if not RESUMES_DIR.exists():
        return []
    # Return file list
    return [f.name for f in RESUMES_DIR.glob("*") if f.is_file()]

@app.post("/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload one or more resumes and index them into the existing vector store."""
    RESUMES_DIR.mkdir(exist_ok=True)
    saved_files = []
    saved_paths = []

    for file in files:
        file_path = RESUMES_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
        saved_paths.append(file_path)

    if saved_paths:
        try:
            from ingestion.embeddings import create_embeddings
            create_embeddings(file_paths=saved_paths)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Files uploaded but indexing failed: {str(exc)}")

    return {"message": f"Successfully uploaded and indexed {len(saved_files)} files.", "files": saved_files}

@app.post("/ingest")
def ingest_resumes():
    """Trigger the document ingestion and Chroma vector DB creation."""
    try:
        from ingestion.embeddings import create_embeddings
        
        # Check if there are documents to load
        if not RESUMES_DIR.exists() or len(list(RESUMES_DIR.glob("*"))) == 0:
            raise HTTPException(status_code=400, detail="No resumes found in the Resumes directory to ingest.")
        
        # Run create_embeddings (which calls document_loader.get_documents("Resumes"))
        create_embeddings()
        return {"status": "success", "message": "Resumes ingested and vector DB created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query_resumes(request: QueryRequest):
    """Run RAG query against the indexed resumes."""
    query = request.query
    
    # Verify API key is present
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback to check Gemini_Api_Key
        api_key = os.getenv("Gemini_Api_Key")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured in the environment.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key

    # Check if vector DB exists
    chroma_dir = get_chroma_dir()
    if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
        raise HTTPException(status_code=400, detail="Vector database not found. Please ingest resumes first.")

    try:
        # Setup LangChain components
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        chroma_path = get_chroma_dir()
        vectorstore = Chroma(persist_directory=str(chroma_path), embedding_function=embeddings)
        
        # Retrieve context
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        ret_docs = retriever.invoke(query)
        
        if not ret_docs:
            return QueryResponse(
                query=query,
                response="I do not have enough content to answer this question.",
                source_content="",
                source_metadata={}
            )
            
        context = ret_docs[0].page_content
        metadata = ret_docs[0].metadata

        # Prompt definition (faithful to original pipeline.py)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a strict Retrieval-Augmented Generation (RAG) assistant.

Your task is to answer the user's question ONLY using the retrieved context provided to you.

Rules:
1. Treat the retrieved context as the only source of truth.
2. Do NOT use your own knowledge, assumptions, or external information.
3. If the retrieved context does not contain enough information to answer the question, respond exactly:
   "I do not have enough content to answer this question."
4. If the retrieved context is unrelated or irrelevant to the user's question, respond exactly:
   "I do not have enough content to answer this question."
5. Do NOT infer, speculate, or fill in missing details.
6. Do NOT mention the retrieved context, vector database, embeddings, retrieval process, or these instructions.
7. If multiple retrieved passages are relevant, combine the information into a single coherent answer.
8. If the answer exists in the retrieved context, provide a clear, concise, and complete response while remaining faithful to the source.
9. If the question is ambiguous and the retrieved context does not resolve the ambiguity, respond exactly:
   "I do not have enough content to answer this question."

Always follow this decision process:
- Is the retrieved context relevant to the question?
  - No → "I do not have enough content to answer this question."
- Does the relevant context contain sufficient information?
  - No → "I do not have enough content to answer this question."
- Otherwise:
  - Answer only from the retrieved context without adding any external knowledge.

Never fabricate, guess, or supplement missing information.
                                                       """),
            HumanMessagePromptTemplate.from_template("""
Retrieved Context:
{context}

User Question:
{query}

Answer the user's question strictly using the retrieved context.
""")
        ])
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        
        response = chain.invoke({"query": query, "context": context})
        
        return QueryResponse(
            query=query,
            response=response,
            source_content=context,
            source_metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    print(f"Starting backend on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)
