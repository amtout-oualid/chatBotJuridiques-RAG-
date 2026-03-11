from pathlib import Path
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
EMB_TEXT_DIR = BASE_DIR / "artifacts" / "embeddings_patch_text"

DOC_NAME = "AMTJ-parents"
PAGE_NAME = "page_001"
TOP_K = 10


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def embed_query(text):
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


def load_text_embeddings():
    data = torch.load(EMB_TEXT_DIR / DOC_NAME / f"{PAGE_NAME}_text.pt")
    return data["embeddings"], data["coords"], data["texts"]


def topk_search(query_emb, patch_embs, k):
    sims = torch.matmul(query_emb.unsqueeze(0), patch_embs.T).squeeze(0)
    return torch.topk(sims, k)


if __name__ == "__main__":
    query = "آجال التسجيل"

    print(f" Requête : {query}")
    q_emb = embed_query(query)

    patch_embs, coords, texts = load_text_embeddings()

    scores, indices = topk_search(q_emb, patch_embs, TOP_K)

    print("\n Résultats Top-K patches (TEXTUELS) :")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"\n#{rank}")
        print(f"Score : {score.item():.4f}")
        print(f"Coords : {coords[idx]}")
        print(f"Texte OCR : {texts[idx]}")
