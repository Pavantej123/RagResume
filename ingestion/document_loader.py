from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
import os


def get_documents(folder_path="Resumes"):
    # document loader
    documents = []
    for i in os.listdir(folder_path):
        path = "Resumes/"
        loader = PyMuPDFLoader(path+i,mode="single")
        document = loader.load()
        documents.extend(document)
    return documents