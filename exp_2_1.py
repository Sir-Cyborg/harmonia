import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

def dist_coseno(embeddings, ID_1, ID_2, n1, soglia=0.3):
    """
    Trova tutte le coppie (ID_1, ID_2) la cui distanza del coseno è minore della soglia.
    """
    risultati = []
    emb1 = embeddings[:n1]
    emb2 = embeddings[n1:]
    dist_matrix = cosine_distances(emb1, emb2)
    for i in range(len(ID_1)):
        for j in range(len(ID_2)):
            dist = dist_matrix[i, j]
            if dist < soglia:
                risultati.append({
                    "ID_F1": ID_1[i],
                    "ID_F2": ID_2[j],
                    "distanza_coseno": dist
                })
    return risultati


with open("F1.json", "r", encoding="utf-8") as f1:
    elementi_1 = json.load(f1)

with open("F2.json", "r", encoding="utf-8") as f2:
    elementi_2 = json.load(f2)

text_1 = [e["TEXT"] for e in elementi_1]
ID_1 = [e["ID"] for e in elementi_1]
text_2 = [e["TEXT"] for e in elementi_2]
ID_2 = [e["ID"] for e in elementi_2]

text=text_1 + text_2
ID=ID_1 + ID_2

# Embedding con BERT
model = SentenceTransformer("bert-base-nli-mean-tokens")
embeddings = model.encode(text)

# PCA a 2D s
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
n1=len(text_1)

# primo set di dati
plt.scatter(embeddings_2d[:n1, 0], embeddings_2d[:n1, 1], c="skyblue", edgecolors="k", label="F1.json")
# secondo set di dati
plt.scatter(embeddings_2d[n1:, 0], embeddings_2d[n1:, 1], c="orange", edgecolors="k", label="F2.json")
plt.legend()

# Aggiungi etichette
for i, id_val in enumerate(ID):
    plt.text(embeddings_2d[i, 0] + 0.05, embeddings_2d[i, 1], id_val, fontsize=8)    

plt.title("Distribuzione dei controlli nello spazio semantico (BERT + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# Dopo aver calcolato embeddings e n1
coppie_vicine = dist_coseno(embeddings, ID_1, ID_2, n1, soglia=0.3)

# Mostra la tabella
df = pd.DataFrame(coppie_vicine)
print(df)
df.to_excel("result.xlsx", index=False)