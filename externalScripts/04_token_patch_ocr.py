from pathlib import Path
import torch
from PIL import Image
import pytesseract
import json

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "data" / "pages" / "AMTJ-parents"
PATCH_SIZE = 64

def extract_patches(img, patch_size=PATCH_SIZE):
    w, h = img.size
    patches = []
    coords = []
    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            box = (left, top, min(left + patch_size, w), min(top + patch_size, h))
            patch = img.crop(box)
            patches.append(patch)
            coords.append(box)
    return patches, coords

def ocr_patches(patches):
    texts = []
    for patch in patches:
        text = pytesseract.image_to_string(patch, lang='ara')
        texts.append(text.strip())
    return texts

def main():
    out_dir = BASE_DIR / "artifacts" / "ocr"
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(PAGES_DIR.glob("*.png"))
    print(f"️ {len(pages)} pages à traiter avec OCR patch-level")

    for img_path in pages:
        print(f"Traitement : {img_path.name}")
        img = Image.open(img_path).convert("RGB")

        patches, coords = extract_patches(img, PATCH_SIZE)
        ocr_texts = ocr_patches(patches)

        ocr_data = []
        for i, (bbox, text) in enumerate(zip(coords, ocr_texts)):
            ocr_data.append({
                "patch_index": i,
                "bbox": bbox,
                "text": text
            })

        json_path = out_dir / f"{img_path.stem}_ocr_patch.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=4)

        print(f" OCR patch-level sauvegardé : {json_path}")

    print(" OCR patch-level terminé sur toutes les pages")

if __name__ == "__main__":
    main()
