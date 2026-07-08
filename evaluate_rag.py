import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
except Exception:
    from datasets import Dataset
    import numpy as np

    def evaluate(df, metrics=None):
        print("RAGAS is unavailable with the current dependency stack; falling back to a simple placeholder evaluation.")
        return pd.DataFrame({
            "faithfulness": [np.nan],
            "answer_relevancy": [np.nan],
            "context_precision": [np.nan],
            "context_recall": [np.nan],
        })

    faithfulness = answer_relevancy = context_precision = context_recall = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "Rag_Resumes (Responses).xlsx"
CHROMA_DIR = BASE_DIR / "chroma.db"

if not EXCEL_PATH.exists():
    raise FileNotFoundError(f"Evaluation sheet not found: {EXCEL_PATH}")

# Read the sheet and normalize expected column names
sheet = pd.read_excel(EXCEL_PATH)

# Map the workbook's actual column names to the fields expected by the evaluator.
query_col = None
truth_col = None
for candidate in ["sample_query", "Sample_query", "query", "question", "Sample Query", "Question"]:
    if candidate in sheet.columns:
        query_col = candidate
        break

for candidate in ["groundtruth", "ground_truth", "groundtruth_answer", "GroundTruth_Answer", "expected_answer", "answer", "Ground Truth", "Expected Answer"]:
    if candidate in sheet.columns:
        truth_col = candidate
        break

if query_col is None or truth_col is None:
    raise ValueError(f"Could not find query/ground truth columns in {EXCEL_PATH}. Available columns: {list(sheet.columns)}")

rows = []
for _, row in sheet.iterrows():
    rows.append({
        "question": str(row[query_col]).strip(),
        "ground_truth": str(row[truth_col]).strip(),
    })

# Ensure a vector store exists; if not, build it from the resumes folder.
if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    from ingestion.embeddings import create_embeddings
    create_embeddings()

# Build the RAG pipeline
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are a strict RAG assistant. Answer the user's question only using the retrieved context.
If the context is insufficient, say exactly: I do not have enough content to answer this question.
"""),
    HumanMessagePromptTemplate.from_template("""
Retrieved Context:
{context}

User Question:
{query}

Answer using only the retrieved context.
""")
])
chain = prompt | llm | StrOutputParser()


def run_rag(question: str):
    docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    answer = chain.invoke({"query": question, "context": context_text})
    return answer, [doc.page_content for doc in docs], [doc.metadata for doc in docs]

results = []
errors = []
for item in rows:
    try:
        answer, contexts, metadatas = run_rag(item["question"])
        results.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
    except Exception as exc:
        errors.append({
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "error": str(exc),
        })
        results.append({
            "question": item["question"],
            "answer": "",
            "contexts": [],
            "ground_truth": item["ground_truth"],
        })

# Create a DataFrame in the format expected by RAGAS
ragas_df = pd.DataFrame(results)
ragas_df = ragas_df[["question", "answer", "contexts", "ground_truth"]]

# Evaluate using RAGAS metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
try:
    score_df = evaluate(ragas_df, metrics=metrics)
except Exception as exc:
    print(f"RAGAS evaluation could not be completed: {exc}")
    score_df = pd.DataFrame({
        "faithfulness": [None],
        "answer_relevancy": [None],
        "context_precision": [None],
        "context_recall": [None],
    })

print("RAGAS Evaluation Results")
print("=" * 40)
print(score_df)

output_csv = BASE_DIR / "ragas_evaluation_results.csv"
score_df.to_csv(output_csv, index=False)
print(f"\nSaved detailed results to {output_csv}")

if errors:
    error_csv = BASE_DIR / "ragas_evaluation_errors.csv"
    pd.DataFrame(errors).to_csv(error_csv, index=False)
    print(f"Saved evaluation errors to {error_csv}")
