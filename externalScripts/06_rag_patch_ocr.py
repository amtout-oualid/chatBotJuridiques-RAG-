

from pathlib import Path
import json
import torch
import torch.nn.functional as F
import clip
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent

EMB_PATCH_DIR = BASE_DIR / "artifacts" / "embeddings_patch"
OCR_DIR = BASE_DIR / "artifacts" / "ocr"

DOC_NAME = "AMTJ-parents"
PAGE_NAME = "page_001"
TOP_K = 5

MODEL_NAME = "gpt-4o-mini"

device = "cuda" if torch.cuda.is_available() else "cpu"


client = OpenAI()


print("Chargement du modèle CLIP...")
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()



def embed_query(text: str) -> torch.Tensor:
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)  # [512]



def load_patch_embeddings(doc_name, page_name):
    path = EMB_PATCH_DIR / doc_name / f"{page_name}_patches.pt"
    data = torch.load(path, map_location=device)

    embeddings = data["embeddings"]  # [N, 512]
    coords = data["coords"]

    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings, coords



def load_ocr_text(doc_name, page_name):
    ocr_path = OCR_DIR / doc_name / f"{page_name}_ocr.json"
    with open(ocr_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d.get("text", "") for d in data]
    return texts



def topk_patches(query_emb, patch_embs, k=TOP_K):
    sims = torch.matmul(query_emb.unsqueeze(0), patch_embs.T).squeeze(0)
    topk_vals, topk_idx = torch.topk(sims, k)
    return (
        topk_vals.detach().cpu().numpy(),
        topk_idx.detach().cpu().numpy(),
    )



if __name__ == "__main__":

    question = "ما هي حقوق الوالدين المذكورة في هذا المستند؟"

    print(f"\n Question : {question}")


    query_emb = embed_query(question)


    print(f" Chargement des patches : {DOC_NAME} / {PAGE_NAME}")
    patch_embs, coords = load_patch_embeddings(DOC_NAME, PAGE_NAME)


    print(f" Recherche Top-{TOP_K} patches...")
    scores, indices = topk_patches(query_emb, patch_embs, TOP_K)


    print(" Chargement OCR...")
    ocr_texts = load_ocr_text(DOC_NAME, PAGE_NAME)


    context_blocks = []
    for rank, idx in enumerate(indices, 1):
        text = ocr_texts[idx] if idx < len(ocr_texts) else ""
        coord = coords[idx]
        score = scores[rank - 1]

        context_blocks.append(
            f"[Patch #{rank} | score={score:.4f} | coords={coord}]\n{text}"
        )

    context_text = "\n\n".join(context_blocks)


    prompt = f"""
أنت مساعد ذكي.
أجب فقط اعتماداً على المعلومات الموجودة في السياق أدناه.
إذا لم توجد المعلومة، قل: "المعلومة غير موجودة في الوثيقة".

السياق:
{context_text}

السؤال:
{question}
"""

    print(" Appel du LLM...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content

    print("\n RÉPONSE :")
    print(answer)
