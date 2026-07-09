import shutil
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingestion.document_loader import get_documents


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma.db"


# embeddings and vectorstore
def create_embeddings(file_paths=None, force_rebuild=False):
    if file_paths is None:
        documents = get_documents(str(BASE_DIR / "Resumes"))
    else:
        documents = get_documents(str(BASE_DIR / "Resumes"), filenames=[str(Path(p).name) for p in file_paths])

    if not documents:
        return CHROMA_DIR

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if force_rebuild and CHROMA_DIR.exists():
        for item in CHROMA_DIR.iterdir():
            if item.is_file() or item.is_dir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        vectorstore.add_documents(documents=documents)
    else:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return CHROMA_DIR
    