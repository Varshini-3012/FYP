import pandas as pd

metadata = pd.read_excel(
    r"C:\Users\varsh\OneDrive\Documents\Datasets_for_autoimmune\Metadata.xlsx"
)

from huggingface_hub import list_repo_files

files = list_repo_files(
    "TokyoTechMagicYang/RAM-H1200-v1",
    repo_type="dataset"
)

split_bases_dict = {}

for split in ["train", "val", "test"]:
    split_stems = [
        f.split("/")[-1].replace(".bmp", "")
        for f in files
        if f.startswith(f"Segmentation/{split}/")
        and f.lower().endswith(".bmp")
    ]

    split_bases = set(
        stem.rsplit("_", 1)[0]
        for stem in split_stems
    )

    split_bases_dict[split] = split_bases

    split_metadata = metadata[
        metadata["Mapped Image Stem"].isin(split_bases)
    ].copy()

    print("\n" + "=" * 50)
    print(split.upper())
    print("Cases:", len(split_metadata))

    print("\nRA distribution:")
    print(split_metadata["isRA"].value_counts())

    print("\nRA percentage:")
    print(
        (split_metadata["isRA"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nSex distribution:")
    print(split_metadata["Sex"].value_counts())