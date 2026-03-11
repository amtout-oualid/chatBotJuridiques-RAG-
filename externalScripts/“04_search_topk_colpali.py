
from pathlib import Path
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
EMB_PATCH_DIR = BASE_DIR / "artifacts/embeddings_patch_colpali"
DOC_NAME = "AMTJ-parents"       # Document à rechercher
PAGE_NAME = "page_001"          # Page à rechercher
TOP_K = 10                      # Nombre de patches à récupérer


print("Chargement du modèle textuel pour embedding...")
model_text = SentenceTransformer('all-MiniLM-L6-v2')

def embed_query(text):
    emb = model_text.encode(text, convert_to_tensor=True)
    return F.normalize(emb, dim=0)


def load_patch_embeddings(doc_name, page_name):
    path = EMB_PATCH_DIR / doc_name / f"{page_name}_patches.pt"
    data = torch.load(path)
    embeddings = data["embeddings"]      # [num_patches x 512]
    coords = data["coords"]              # [(left, top, right, bottom)]
    ocr_texts = data["ocr_texts"]        # texte OCR patch-level
    embeddings = F.normalize(embeddings, dim=1)
    return embeddings, coords, ocr_texts


def topk_patches(query_emb, patch_embs, k=10):
    sims = torch.matmul(query_emb.unsqueeze(0), patch_embs.T).squeeze(0)
    topk_vals, topk_idx = torch.topk(sims, k)
    return topk_vals.detach().cpu().numpy(), topk_idx.detach().cpu().numpy()


if __name__ == "__main__":

    question = "موعد التسجيل"

    print(f"Embedding de la requête : '{question}'")
    query_emb = embed_query(question)

    print(f"Chargement des embeddings patches pour {DOC_NAME} {PAGE_NAME}")
    patch_embs, coords, ocr_texts = load_patch_embeddings(DOC_NAME, PAGE_NAME)

    print(f"Calcul top-{TOP_K} patches les plus proches...")
    topk_scores, topk_indices = topk_patches(query_emb, patch_embs, TOP_K)

    print("\nRésultats Top-K patches :")
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), 1):
        coord = coords[idx]
        text = ocr_texts[idx]
        print(f"#{rank:02d} patch idx={idx} score={score:.4f} coords={coord}")
        print(f"    OCR text: {text[:50]}{'...' if len(text)>50 else ''}")
