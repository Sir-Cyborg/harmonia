import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with open("controlli.json", "r", encoding="utf-8") as f1:
    controlli1 = json.load(f1)
with open("controlli_2.json", "r", encoding="utf-8") as f2:
    controlli2 = json.load(f2)

descrizioni1 = [c["descrizione"] for c in controlli1]
titoli1 = [c["titolo"] for c in controlli1]
descrizioni2 = [c["descrizione"] for c in controlli2]
titoli2 = [c["titolo"] for c in controlli2]

descrizioni = descrizioni1 + descrizioni2
titoli = titoli1 + titoli2

model = SentenceTransformer("bert-base-nli-mean-tokens")
embeddings = model.encode(descrizioni)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Step 5: Visualizza
plt.figure(figsize=(12, 8))
n1 = len(descrizioni1)
n2 = len(descrizioni2)

# Primo gruppo: blu
plt.scatter(embeddings_2d[:n1, 0], embeddings_2d[:n1, 1], c="skyblue", edgecolors="k", label="controlli_1.json")
# Secondo gruppo: arancione
plt.scatter(embeddings_2d[n1:, 0], embeddings_2d[n1:, 1], c="orange", edgecolors="k", label="controlli_2.json")

# Aggiungi etichette
for i, titolo in enumerate(titoli):
    plt.text(embeddings_2d[i, 0] + 0.5, embeddings_2d[i, 1], titolo, fontsize=8)

plt.title("Distribuzione dei controlli nello spazio semantico (BERT + PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend()
plt.show()