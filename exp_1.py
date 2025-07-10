import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


with open("F1.json", "r", encoding="utf-8") as f:
    elemento = json.load(f)


text = [e["TEXT"] for e in elemento]
ID = [e["ID"] for e in elemento]

# Embedding con BERT
#"all-MiniLM-L6-v2"
model = SentenceTransformer("bert-base-nli-mean-tokens")
embeddings = model.encode(text)

# PCA a 2D s
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="skyblue", edgecolors="k")

# Aggiungi etichette
for i, id_val in enumerate(ID):
    plt.text(embeddings_2d[i, 0] + 0.05, embeddings_2d[i, 1], id_val, fontsize=8)    

plt.title("Distribuzione dei controlli nello spazio semantico (BERT + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
