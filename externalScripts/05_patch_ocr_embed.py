from pathlib import Path
import torch
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
PATCH_IMG_DIR = BASE_DIR / "artifacts" / "patches"
OUT_DIR = BASE_DIR / "artifacts" / "embeddings_patch_text"

PATCH_SIZE = 64
OCR_LANG = "ara"


model_text = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def ocr_patch(patch_img):
    text = pytesseract.image_to_string(patch_img, lang=OCR_LANG)
    return text.strip()


for doc_dir in PATCH_IMG_DIR.iterdir():
    if not doc_dir.is_dir():
        continue

    print(f" Document : {doc_dir.name}")
    doc_out = OUT_DIR / doc_dir.name
    doc_out.mkdir(parents=True, exist_ok=True)

    for page_dir in doc_dir.iterdir():
        if not page_dir.is_dir():
            continue

        print(f"   Page : {page_dir.name}")
        texts = []
        coords = []

        for patch_path in sorted(page_dir.glob("*.png")):
            img = Image.open(patch_path)
            text = ocr_patch(img)

            if text == "":
                continue

            texts.append(text)


            name = patch_path.stem
            left = int(name.split("_x")[1].split("_")[0])
            top = int(name.split("_y")[1])
            coords.append((left, top, left + PATCH_SIZE, top + PATCH_SIZE))

        if not texts:
            print("    Aucun texte détecté")
            continue

        embeddings = model_text.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

        torch.save(
            {
                "embeddings": embeddings,
                "coords": coords,
                "texts": texts
            },
            doc_out / f"{page_dir.name}_text.pt"
        )

        print(f"     {len(texts)} patches OCR + embeddings sauvegardés")

print(" OCR + embeddings textuels terminés")
