import streamlit as st

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

# Title
st.title("AI-Docs Retriever")

# Search bar input
query = st.text_input("Enter keywords to search:")

# Search button
if st.button("Search"):
    if query:
        st.write(f"Searching for: {query}")
        # Placeholder for search logic
    else:
        st.warning("Please enter some keywords to search.")
