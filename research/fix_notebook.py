import json

file_path = "c:/Users/vivobook/Desktop/INPT/Me/Project/developpementProject/chatBotJuridiques-RAG-/research/trials.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if any("from pathlib import Path\n" in line for line in source) and any("process_all_pdfs()" in line for line in source):
            # Replace the base dir logic
            new_source = []
            for line in source:
                if "current_path = Path.cwd()" in line:
                    # Replace with a more robust way to find the base dir
                    new_source.append("import os\n")
                    new_source.append("# Find the project root by looking for 'data' folder\n")
                    new_source.append("current_path = Path(os.path.abspath(''))\n")
                    new_source.append("while current_path.name != 'chatBotJuridiques-RAG-' and current_path.parent != current_path:\n")
                    new_source.append("    current_path = current_path.parent\n")
                elif "BASE_DIR = current_path.parent" in line:
                    new_source.append("BASE_DIR = current_path  # The root of the project\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    # add a newline at end to match typical ipynb formatting
    f.write("\n")

print("Updated trials.ipynb")
