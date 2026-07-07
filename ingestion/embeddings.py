from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingestion.document_loader import get_documents



#embeddings and vectorstore
def create_embeddings():
    documents = get_documents()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(documents=documents,embedding=embeddings,persist_directory="chroma.db")
    