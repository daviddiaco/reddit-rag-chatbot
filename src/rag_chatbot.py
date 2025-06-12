import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/faiss_index.pkl", "rb") as f:
    index = pickle.load(f)

with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

query = "What new token launches are trending?"
query_vec = model.encode([query])
_, I = index.search(np.array(query_vec), k=5)

for i in I[0]:
    print("----")
    print(texts[i][:300])
