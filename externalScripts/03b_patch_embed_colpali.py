

from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18
import pytesseract

BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "data/pages"
OUT_DIR = BASE_DIR / "artifacts/embeddings_patch_colpali"
PATCH_SIZE = 64


model = resnet18(weights=None)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier

transform = T.Compose([T.ToTensor()])

OUT_DIR.mkdir(parents=True, exist_ok=True)


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


for doc_dir in PAGES_DIR.iterdir():
    if not doc_dir.is_dir():
        continue

    print(f"Document : {doc_dir.name}")
    doc_out = OUT_DIR / doc_dir.name
    doc_out.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(doc_dir.glob("*.png")):
        print(f"  Page : {img_path.name}")
        img = Image.open(img_path).convert("RGB")


        patches, coords = extract_patches(img, PATCH_SIZE)


        embeddings = []
        for patch in patches:
            x = transform(patch).unsqueeze(0)
            with torch.no_grad():
                emb = model(x).squeeze()
            embeddings.append(emb)
        embeddings_tensor = torch.stack(embeddings)


        ocr_texts = ocr_patches(patches)


        torch.save({
            "embeddings": embeddings_tensor,
            "coords": coords,
            "ocr_texts": ocr_texts,
            "page": img_path.name,
            "doc": doc_dir.name
        }, doc_out / f"{img_path.stem}_patches.pt")

        print(f"    {len(patches)} patches traités et sauvegardés avec OCR")

print(" Embeddings patch-level ColPali simulés terminés")
