"""Load data and index."""

import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_chunks(pickle_path):
    """Load serialised text chunks."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_faiss_index(index_path):
    """Load FAISS index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
