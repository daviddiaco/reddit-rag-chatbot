import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle


# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load posts
def load_posts(path="data/posts.json"):
    with open(path, "r") as f:
        return json.load(f)

# Build the FAISS index
def build_index(posts):
    texts = [p['title'] + "\n" + p['selftext'] for p in posts]
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

# Save the index
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Main script
if __name__ == "__main__":
    posts = load_posts()
    index, texts = build_index(posts)
    save_pickle(index, "data/faiss_index.pkl")
    save_pickle(texts, "data/texts.pkl")
    print("Index and texts saved as pickle.")

