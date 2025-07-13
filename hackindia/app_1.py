import streamlit as st
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(page_title="AI-Docs Retriever", page_icon="📄", layout="wide")

# Custom CSS Styling
st.markdown(
    """
    <style>
        /* Background Styling */
        .stApp {
            background-color: #F8FAFC;
        }

        /* Title Styling */
        h1 {
            color: #002855;
            text-align: center;
            font-size: 38px;
            font-weight: bold;
        }

        /* Search Box Styling */
        .stTextInput > div > div > input {
            background-color: white;
            color: #002855;
            border-radius: 12px;
            border: 2px solid #4CAF50;
            font-size: 18px;
            padding: 10px;
        }

        /* Search Button Styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            width: 100%;
            padding: 10px;
        }

        /* Hover Effect for Button */
        .stButton>button:hover {
            background-color: #3e8e41;
            transition: 0.3s ease-in-out;
            transform: scale(1.05);
        }

        /* Document Results Styling */
        .doc-card {
            background-color: white;
            padding: 12px;
            margin: 8px 0;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: 0.3s ease-in-out;
        }

        .doc-card:hover {
            transform: scale(1.02);
            background-color: #f1f8e9;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index and document metadata
faiss_index = faiss.read_index("faiss_index.bin")
doc_vectors = np.load("doc_vectors.npy")
with open("doc_filenames.json", "r") as f:
    doc_filenames = json.load(f)

# Main Title
st.title("📄 AI-Docs Retriever")

# Centered Search Box
query = st.text_input("🔍 **Enter keywords to search:**", key="search_input")

# Search Button
col1, col2, col3 = st.columns([2, 1, 2])  # Centering button
with col2:
    search_clicked = st.button("🔎 Search")

# Search Logic
if search_clicked:
    if query:
        st.success(f"🔍 **Searching for:** {query}")

        # Convert query to vector
        query_vector = model.encode([query])

        # Perform FAISS search
        k = 5  # Number of results
        distances, indices = faiss_index.search(query_vector, k)

        # Display results
        st.markdown("## 📂 **Top Matching Documents:**")
        for idx, doc_idx in enumerate(indices[0]):
            st.markdown(f"""
            <div class="doc-card">
                <b>📌 {idx+1}. {doc_filenames[doc_idx]}</b> <br>
                <small>🔹 Similarity Score: {distances[0][idx]:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ **Please enter some keywords to search.**")
