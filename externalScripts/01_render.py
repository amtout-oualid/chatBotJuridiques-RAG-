from pathlib import Path
from pdf2image import convert_from_path
PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pages"
DPI = 300
POPPLER_PATH = r"C:\poppler\Library\bin"
print(f"DEBUG: dossier PDF = {PDF_DIR.resolve()}")
print(f"DEBUG: dossier sortie = {OUT_DIR.resolve()}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pdfs = list(PDF_DIR.glob("*.pdf"))
print(f"DEBUG: nombre de fichiers PDF trouvés : {len(pdfs)}")
if not pdfs:
    print(" Aucun PDF trouvé dans data/pdfs")
    raise SystemExit()
for pdf_path in pdfs:
    print(f"DEBUG: traitement du PDF {pdf_path.name}")
    doc_id = pdf_path.stem
    doc_out = OUT_DIR / doc_id
    doc_out.mkdir(parents=True, exist_ok=True)
    try:
        pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            fmt="png",
            poppler_path=POPPLER_PATH
        )
        print(f"DEBUG: {len(pages)} pages converties")
    except Exception as e:
        print(f"Erreur lors de la conversion du PDF {pdf_path.name} : {e}")
        continue
    for i, page in enumerate(pages, start=1):
        page_path = doc_out / f"page_{i:03d}.png"
        page.save(page_path, "PNG")
        print(f"DEBUG: page sauvegardée -> {page_path}")
    print(f"✔ {pdf_path.name} → {len(pages)} pages rendues")
