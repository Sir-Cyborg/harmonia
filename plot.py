from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessario per 3D
import numpy as np

def plot_embeddings_3d(embeddings_A, embeddings_B, ids_A, ids_B, label_A="F1.json", label_B="F2.json"):
    """
    Visualizza in 3D (PCA) due insiemi di embedding con le rispettive etichette ID.
    """
    # Unisci embeddings e ID
    all_embeddings = np.concatenate([embeddings_A, embeddings_B], axis=0)
    all_ids = ids_A + ids_B
    n1 = len(ids_A)

    # PCA a 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Primo set di dati
    ax.scatter(
        embeddings_3d[:n1, 0], embeddings_3d[:n1, 1], embeddings_3d[:n1, 2],
        c="skyblue", edgecolors="k", label=label_A
    )
    # Secondo set di dati
    ax.scatter(
        embeddings_3d[n1:, 0], embeddings_3d[n1:, 1], embeddings_3d[n1:, 2],
        c="orange", edgecolors="k", label=label_B
    )
    ax.legend()

    # Aggiungi etichette ID
    for i, id_val in enumerate(all_ids):
        ax.text(
            embeddings_3d[i, 0] + 0.05,
            embeddings_3d[i, 1],
            embeddings_3d[i, 2],
            str(id_val),
            fontsize=8
        )

    ax.set_title("Distribuzione dei controlli nello spazio semantico (BERT + PCA 3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    return fig