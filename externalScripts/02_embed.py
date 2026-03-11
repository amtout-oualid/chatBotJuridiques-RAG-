from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18

print(" EMBEDDING VISUEL – PATCH LEVEL")


BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = BASE_DIR / "data" / "pages"
OUT_DIR = BASE_DIR / "artifacts" / "embeddings"


model = resnet18(weights=None)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

for doc_dir in PAGES_DIR.iterdir():
    if not doc_dir.is_dir():
        continue

    print(f" Document : {doc_dir.name}")
    doc_out = OUT_DIR / doc_dir.name
    doc_out.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(doc_dir.glob("*.png")):
        print(f"   Page : {img_path.name}")

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            emb = model(x).squeeze()

        torch.save(
            {
                "embedding": emb,
                "page": img_path.name,
                "doc": doc_dir.name
            },
            doc_out / f"{img_path.stem}.pt"
        )

        print(f"    ✔ sauvegardé → {doc_out / f'{img_path.stem}.pt'}")

print("✅ Embeddings terminés")
