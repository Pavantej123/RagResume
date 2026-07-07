from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path


def get_documents(path_folder):
    # 1. Define the relative path to your folder

    folder_path = Path(path_folder)
    # 2. Extract all PDF file paths from the folder
    pdf_paths = [file for file in folder_path.glob("*.pdf") if file.is_file()]

    # 3. Print the paths to verify they are found
    print("Found PDFs:", pdf_paths)

    # 4. Loop through the paths and load them with LangChain
    all_documents = []
    for pdf_path in pdf_paths:
        # PyMuPDFLoader requires a string path, so we use str(pdf_path)
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents
