import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from plot import plot_embeddings_3d

st.title("Dashboard Similarità Testi")

# Caricamento file
uploaded_file_A = st.file_uploader("Carica F1.json", type="json")
uploaded_file_B = st.file_uploader("Carica F2.json", type="json")

if uploaded_file_A and uploaded_file_B:
    # Carica JSON
    data_A = json.load(uploaded_file_A)
    data_B = json.load(uploaded_file_B)

    ids_A = [item["ID"] for item in data_A]
    texts_A = [item["TEXT"] for item in data_A]
    ids_B = [item["ID"] for item in data_B]
    texts_B = [item["TEXT"] for item in data_B]

    st.write("✅ File caricati correttamente!")

    model = SentenceTransformer('all-mpnet-base-v2')
    with st.spinner("Calcolo degli embeddings e delle similarità..."):
        embeddings_A = model.encode(texts_A, convert_to_tensor=True)
        embeddings_B = model.encode(texts_B, convert_to_tensor=True)

        results = []
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

        df = pd.DataFrame(results)

    st.success("Calcolo completato!")

    # Mostra DataFrame
    st.subheader("Risultati delle similarità")
    st.dataframe(df)

    # Mostra grafico
    st.subheader("Distribuzione delle similarità")
    fig=plot_embeddings_3d(embeddings_A, embeddings_B, ids_A, ids_B)
    st.pyplot(fig)

    # Download risultati
    st.subheader("Scarica i risultati")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Scarica CSV", data=csv, file_name="result.csv", mime="text/csv")

else:
    st.info("Carica entrambi i file JSON per iniziare.")
