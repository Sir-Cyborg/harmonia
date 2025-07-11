############################################
#### Visualizzazione 2D dei controlli
### Serve check sulla riduzione dimensione
############################################

from sentence_transformers import SentenceTransformer
import numpy as np
import umap
import matplotlib.pyplot as plt


controls_A = [
    "Implement strong authentication mechanisms for remote access",
    "Define a formal security incident response plan",
    "Encrypt sensitive data at rest and in transit",
    "Regularly update and patch operating systems and applications",
    "Conduct periodic security awareness training for employees"
]

controls_B = [
    "Establish an information security policy approved by management",
    "Ensure appropriate security controls are implemented for remote working",
    "Implement procedures for managing information security incidents",
    "Apply cryptographic controls to protect data",
    "Ensure operating systems are regularly updated and patched",
    "Establish and maintain an information security awareness program",
    "Define and implement an access control policy"
]


model = SentenceTransformer('all-mpnet-base-v2')

embeddings_A = model.encode(controls_A)
embeddings_B = model.encode(controls_B)

# Combina per visualizzare insieme
all_embeddings = np.vstack([embeddings_A, embeddings_B])

# Riduci a 2D
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
embeddings_2d = umap_model.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:len(controls_A), 0], embeddings_2d[:len(controls_A), 1], color='blue', label='Controls A')
plt.scatter(embeddings_2d[len(controls_A):, 0], embeddings_2d[len(controls_A):, 1], color='green', label='Controls B')

for i, text in enumerate(controls_A):
    plt.annotate(f"A{i+1}", (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
for i, text in enumerate(controls_B):
    plt.annotate(f"B{i+1}", (embeddings_2d[len(controls_A) + i, 0], embeddings_2d[len(controls_A) + i, 1]), fontsize=9)

plt.legend()
plt.title("Visualizzazione 2D dei controlli (UMAP)")
plt.show()
