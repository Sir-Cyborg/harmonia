import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


with open("controlli_1.json", "r", encoding="utf-8") as f:
    controlli = json.load(f)


descrizioni = [c["descrizione"] for c in controlli]
titoli = [c["titolo"] for c in controlli]

# Embedding con BERT
model = SentenceTransformer("bert-base-nli-mean-tokens")
embeddings = model.encode(descrizioni)

# PCA a 2D s
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="skyblue", edgecolors="k")

# Aggiungi etichette
for i, titolo in enumerate(titoli):
    plt.text(embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1], titolo, fontsize=8)

plt.title("Distribuzione dei controlli nello spazio semantico (BERT + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
