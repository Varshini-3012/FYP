from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "TokyoTechMagicYang/RAM-H1200-v1"

BASE_DIR = Path(
    r"C:\Users\varsh\OneDrive\Documents\Datasets_for_autoimmune"
)

SEG_DIR = BASE_DIR / "Segmentation"

SEG_DIR.mkdir(parents=True, exist_ok=True)

print("Starting dataset download...")
print("Destination:", SEG_DIR)

files = []

from huggingface_hub import list_repo_files

all_files = list_repo_files(
    REPO_ID,
    repo_type="dataset"
)

for file in all_files:
    if file.startswith("Segmentation/") and file.lower().endswith(".bmp"):
        files.append(file)

print(f"Found {len(files)} segmentation images.")

for i, file in enumerate(files, start=1):

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=file,
        repo_type="dataset",
        local_dir=str(BASE_DIR)
    )

    print(f"[{i}/{len(files)}] Downloaded: {file}")

print("\n========================================")
print("DOWNLOAD COMPLETE")
print("========================================")
print(f"Total files: {len(files)}")
print(f"Location: {SEG_DIR}")