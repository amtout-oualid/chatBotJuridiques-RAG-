from pathlib import Path
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent.parent
EMB_PATCH_DIR = BASE_DIR / "artifacts" / "embeddings_patch"
DOC_NAME = "AMTJ-parents"           # Nom du dossier document
PAGE_NAME = "page_001"              # Page à rechercher (sans extension)
TOP_K = 10                         # Nombre de patches à récupérer


# Charger le modèle sentence-transformers
print("Chargement du modèle textuel pour embedding...")
model_text = SentenceTransformer('all-MiniLM-L6-v2')

# Fonction pour vectoriser la requête
def embed_query(text):
    emb = model_text.encode(text, convert_to_tensor=True)
    return F.normalize(emb, dim=0)


# Charger embeddings patch-level et coordonnées
def load_patch_embeddings(doc_name, page_name):
    path = EMB_PATCH_DIR / doc_name / f"{page_name}_patches.pt"
    data = torch.load(path)
    embeddings = data["embeddings"]
    coords = data["coords"]
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings, coords


# Calculer top-k patches les plus proches
def topk_patches(query_emb, patch_embs, k=10):
    # produit scalaire (cosine similarity)
    sims = torch.matmul(query_emb.unsqueeze(0), patch_embs.T).squeeze(0)
    topk_vals, topk_idx = torch.topk(sims, k)
    return topk_vals.detach().cpu().numpy(), topk_idx.detach().cpu().numpy()



if __name__ == "__main__":
    question = "délai inscription"

    print(f"Embedding de la requête : '{question}'")
    query_emb = embed_query(question)

    proj = nn.Linear(384, 512, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(512, 384))

    query_emb = proj(query_emb.unsqueeze(0)).squeeze(0)
    query_emb = F.normalize(query_emb, dim=0)

    print(f"Chargement des embeddings patches pour {DOC_NAME} {PAGE_NAME}")
    patch_embs, coords = load_patch_embeddings(DOC_NAME, PAGE_NAME)

    print(f"Calcul top-{TOP_K} patches les plus proches...")
    topk_scores, topk_indices = topk_patches(query_emb, patch_embs, TOP_K)

    print("\nRésultats Top-K patches :")
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
        coord = coords[idx]
        print(f"#{rank:02d} patch idx={idx} score={score:.4f} coords={coord}")
