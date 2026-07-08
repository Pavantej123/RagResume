from langchain_core.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.runnables import RunnableSequence
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory="chroma.db",embedding_function=embeddings)
query = input("Ask a question: ")
retriver = vectorstore.as_retriever(search_type = "similarity",search_kwargs={"k":1})
ret_docs = retriver.invoke(query)
if not ret_docs:
    print("I do not have enough content to answer this question.")
    exit()
context = ret_docs[0].page_content

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
output = StrOutputParser()

chain = prompt|llm|output

response = chain.invoke({"query":query,"context":context})
print(response)

