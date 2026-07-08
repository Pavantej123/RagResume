from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import os


def get_documents(folder_path="Resumes", filenames=None):
    documents = []
    folder = Path(folder_path)

    if filenames is None:
        selected_files = [folder / name for name in os.listdir(folder)]
    else:
        selected_files = []
        for item in filenames:
            path = Path(item)
            if not path.is_absolute():
                path = folder / path
            selected_files.append(path)

    for file_path in selected_files:
        if not file_path.exists() or not file_path.is_file():
            continue
        if file_path.suffix.lower() != ".pdf":
            continue
        try:
            loader = PyMuPDFLoader(str(file_path), mode="single")
            document = loader.load()
            documents.extend(document)
        except Exception:
            continue

    return documents