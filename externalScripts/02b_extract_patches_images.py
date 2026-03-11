from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "data" / "pages"
PATCH_OUT_DIR = BASE_DIR / "artifacts" / "patches"

PATCH_SIZE = 64

PATCH_OUT_DIR.mkdir(parents=True, exist_ok=True)

for doc_dir in PAGES_DIR.iterdir():
    if not doc_dir.is_dir():
        continue

    print(f"Document : {doc_dir.name}")
    doc_out = PATCH_OUT_DIR / doc_dir.name
    doc_out.mkdir(parents=True, exist_ok=True)

    for page_path in doc_dir.glob("*.png"):
        page_name = page_path.stem
        print(f"  Page : {page_name}")

        page_out = doc_out / page_name
        page_out.mkdir(parents=True, exist_ok=True)

        img = Image.open(page_path).convert("RGB")
        w, h = img.size

        for top in range(0, h, PATCH_SIZE):
            for left in range(0, w, PATCH_SIZE):
                patch = img.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE))
                patch_path = page_out / f"patch_x{left}_y{top}.png"
                patch.save(patch_path)

        print("    ✔ patches sauvegardés")

print("Extraction des patches terminée")
