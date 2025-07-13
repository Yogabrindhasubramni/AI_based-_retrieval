import streamlit as st
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(page_title="AI-Docs Retriever", page_icon="📄")

# Custom CSS for background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ADD8E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load pre-trained Sentence Transformer model (same used for document embedding)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index and document metadata
faiss_index = faiss.read_index("faiss_index.bin")
doc_vectors = np.load("doc_vectors.npy")  # Document vectors
with open("doc_filenames.json", "r") as f:
    doc_filenames = json.load(f)  # List of filenames

# Title
st.title("📄 AI-Docs Retriever")

# Search bar input
query = st.text_input("🔍 Enter keywords to search:")

# Search button
if st.button("Search"):
    if query:
        st.write(f"🔎 Searching for: **{query}**")

        # ✅ Step 6: Convert User Query to Vector
        query_vector = model.encode([query])  # Convert query into 384-dim vector

        # ✅ Step 7: Perform FAISS Nearest Neighbor Search
        k = 5  # Number of top results to retrieve
        distances, indices = faiss_index.search(query_vector, k)

        # Display the top 5 results
        st.subheader("📂 Top Matching Documents:")
        for idx, doc_idx in enumerate(indices[0]):
            st.write(f"**{idx+1}. {doc_filenames[doc_idx]}** (Similarity Score: {distances[0][idx]:.2f})")

    else:
        st.warning("⚠️ Please enter some keywords to search.")
