import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Per il 3D

# Carica i dati dai file JSON
with open("F1.json", "r", encoding="utf-8") as f1:
    elementi_1 = json.load(f1)

with open("F2.json", "r", encoding="utf-8") as f2:
    elementi_2 = json.load(f2)

text_1 = [e["TEXT"] for e in elementi_1]
ID_1 = [e["ID"] for e in elementi_1]
text_2 = [e["TEXT"] for e in elementi_2]
ID_2 = [e["ID"] for e in elementi_2]

text = text_1 + text_2
ID = ID_1 + ID_2

# Embedding con BERT
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(text)

# PCA a 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
n1 = len(text_1)

# Primo set di dati
ax.scatter(embeddings_3d[:n1, 0], embeddings_3d[:n1, 1], embeddings_3d[:n1, 2], c="skyblue", edgecolors="k", label="F1.json")
# Secondo set di dati
ax.scatter(embeddings_3d[n1:, 0], embeddings_3d[n1:, 1], embeddings_3d[n1:, 2], c="orange", edgecolors="k", label="F2.json")
ax.legend()

# Aggiungi etichette
for i, id_val in enumerate(ID):
    ax.text(embeddings_3d[i, 0] + 0.05, embeddings_3d[i, 1], embeddings_3d[i, 2], id_val, fontsize=8)

ax.set_title("Distribuzione dei controlli nello spazio semantico (BERT + PCA 3D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()