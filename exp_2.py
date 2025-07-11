######################
## Dataset su F1 e F2
######################

import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

with open("F1.json", "r", encoding="utf-8") as f:
    data_A = json.load(f)
with open("F2.json", "r", encoding="utf-8") as f:
    data_B = json.load(f)

ids_A = [item["ID"] for item in data_A]
texts_A = [item["TEXT"] for item in data_A]
ids_B = [item["ID"] for item in data_B]
texts_B = [item["TEXT"] for item in data_B]

model = SentenceTransformer('all-mpnet-base-v2')
embeddings_A = model.encode(texts_A, convert_to_tensor=True)
embeddings_B = model.encode(texts_B, convert_to_tensor=True)

results = []

# Per ciascun controllo di A, trova il più simile in B
for idx_a, emb_a in enumerate(embeddings_A):
    cosine_scores = util.cos_sim(emb_a, embeddings_B)[0]
    best_match_idx = np.argmax(cosine_scores)
    best_score = cosine_scores[best_match_idx].item()
    results.append({
        "ID_A": ids_A[idx_a],
        "Testo_A": texts_A[idx_a],
        "ID_B": ids_B[best_match_idx],
        "Testo_B": texts_B[best_match_idx],
        "Cosine_Similarity": best_score
    })

# Crea DataFrame e salva in Excel
df = pd.DataFrame(results)
print(df.head())  # Mostra le prime righe a terminale
df.to_excel("result.xlsx", index=False)