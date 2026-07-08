import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# App Page Configurations
st.set_page_config(
    page_title="RagResume - AI Resume Search Engine",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = "http://localhost:8000"

# Premium CSS Injection for sleek glassmorphic UI and custom styles
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1e2640 100%);
        color: #e2e8f0;
    }
    
    /* Headers styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #818cf8;
        border-radius: 4px;
        padding: 1rem;
        font-family: monospace;
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    
    /* Interactive Button Tweaks */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%) !important;
    }
    
    /* Sidebar styling override */
    .css-1633qas, [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Custom tag styles for resume list */
    .resume-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        margin: 0.2rem;
        font-size: 0.85rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Helper: Fetch Resumes List
def fetch_resumes():
    try:
        response = requests.get(f"{API_URL}/resumes")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []

# Sidebar Content
with st.sidebar:
    st.markdown("### 💼 Document Management")
    st.markdown("Manage your resume database and ingestion process.")
    st.markdown("---")
    
    # Upload Resumes Section
    st.markdown("#### 📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF resumes to add to the system."
    )
    
    if uploaded_files:
        if st.button("🚀 Upload & Save"):
            files_to_send = []
            for file in uploaded_files:
                files_to_send.append(("files", (file.name, file.read(), "application/pdf")))
            
            with st.spinner("Uploading files..."):
                try:
                    response = requests.post(f"{API_URL}/upload", files=files_to_send)
                    if response.status_code == 200:
                        st.success(f"Uploaded {len(uploaded_files)} resumes successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

    st.markdown("---")
    
    # Ingestion Actions
    st.markdown("#### ⚙️ Vector Indexing")
    if st.button("🏗️ Rebuild Vector DB"):
        with st.spinner("Processing resumes and building index..."):
            try:
                response = requests.post(f"{API_URL}/ingest")
                if response.status_code == 200:
                    st.success("Vector DB rebuilt successfully!")
                else:
                    st.error(f"Ingestion failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

    st.markdown("---")

    # List Current Resumes
    st.markdown("#### 📂 Indexed Resumes")
    resumes_list = fetch_resumes()
    if resumes_list:
        st.markdown(f"**Total Resumes:** {len(resumes_list)}")
        for resume in resumes_list:
            st.markdown(f"<div class='resume-tag'>📄 {resume}</div>", unsafe_allow_html=True)
    else:
        st.info("No resumes uploaded or backend is offline.")

# Main Page Content
st.markdown("<h1>RagResume Search Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Retrieval-Augmented Generation (RAG) assistant for querying applicant resumes.</p>", unsafe_allow_html=True)

# Main query panel
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
query_text = st.text_input(
    "Ask a question about the candidate resumes:",
    placeholder="e.g., Which candidates have experience with React and Node.js?",
    key="query_input"
)

if st.button("🔍 Search & Analyze") or (query_text and st.session_state.get("run_search", False)):
    if not query_text.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving resume content and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query_text}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("### 🤖 Answer:")
                    # Display the answer in a beautiful markdown format
                    st.info(result["response"])
                    
                    # If source content is returned, display it
                    if result["source_content"]:
                        st.markdown("---")
                        with st.expander("📚 View Retrieved Source Context"):
                            source_meta = result.get("source_metadata", {})
                            source_file = source_meta.get("source", "Unknown file")
                            # Strip path info from filename
                            source_filename = os.path.basename(source_file)
                            page_num = source_meta.get("page", 0) + 1
                            
                            st.markdown(f"**Source Document:** `{source_filename}` (Page {page_num})")
                            st.markdown(f"<div class='source-card'>{result['source_content']}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Error from server: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to query backend server: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# Helpful guide footer
st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3>1. Upload Resumes</h3>
            <p>Drag and drop PDF resumes in the sidebar. We automatically copy the initial 27 resumes on startup.</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3>2. Index Vector DB</h3>
            <p>Click "Rebuild Vector DB" to extract PDF text, generate embeddings, and build the Chroma database.</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3>3. Strict RAG Answers</h3>
            <p>Ask search queries. The engine answers strictly from the resumes, avoiding hallucination.</p>
        </div>
    """, unsafe_allow_html=True)
