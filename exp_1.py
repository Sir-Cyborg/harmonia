#################################################################
##### Mappa A in B    
#################################################################

from sentence_transformers import SentenceTransformer, util
import numpy as np

controls_A = [
    "Implement strong authentication mechanisms for remote access",
    "Define a formal security incident response plan",
    "Encrypt sensitive data at rest and in transit",
    "Update and patch operating systems and applications",
    "Conduct periodic security awareness training for employees",
    "The cat is on the roof"
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

embeddings_A = model.encode(controls_A, convert_to_tensor=True)
embeddings_B = model.encode(controls_B, convert_to_tensor=True)

# Per ciascun controllo di A, trova il più simile in B
for idx_a, emb_a in enumerate(embeddings_A):
    cosine_scores = util.cos_sim(emb_a, embeddings_B)[0]
    best_match_idx = np.argmax(cosine_scores)
    best_score = cosine_scores[best_match_idx]
    
    print(f"\nControl A: {controls_A[idx_a]}")
    print(f"Best match in B: {controls_B[best_match_idx]}")
    print(f"Cosine similarity score: {best_score:.4f}")

