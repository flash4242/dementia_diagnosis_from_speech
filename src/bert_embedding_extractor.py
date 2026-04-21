import os
import glob
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DATA_DIR = "./data"
OUTPUT_CSV = "./csv_output/bert_embeddings.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Load BERT
# ----------------------------
def load_bert():
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ----------------------------
# Extract text from CHA file
# ----------------------------
def read_cha_file(path):
    """
    Extract spoken utterances from CHAT (.cha) transcript.
    Keeps only participant speech lines (*).
    """

    utterances = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # CHAT speech lines start with *
            if line.startswith("*PAR"):
                # remove speaker tag (*PAR:, *INV:, etc.)
                text = line.split(":", 1)[-1]
                utterances.append(text)

    return " ".join(utterances)

def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    # A split helyett csak a .cha kiterjesztést vágjuk le!
    # Így a 002-1.cha-ból 002-1 marad (String formátumban).
    patient_id = os.path.splitext(filename)[0]
    return patient_id
# ----------------------------
# BERT Embedding
# ----------------------------
def get_bert_embedding(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().cpu().numpy()


# ----------------------------
# Main Extraction Pipeline
# ----------------------------
def extract_features():

    tokenizer, model = load_bert()

    cha_files = glob.glob(
        os.path.join(DATA_DIR, "**/*.cha"),
        recursive=True
    )

    print(f"Found {len(cha_files)} .cha files.")

    dataset_rows = []

    for file_path in tqdm(cha_files):

        if "dementia-trans" in file_path:
            label = 1
        elif "control-trans" in file_path:
            label = 0
        else:
            continue

        try:
            text = read_cha_file(file_path)

            if len(text.strip()) == 0:
                continue

            embedding = get_bert_embedding(text, tokenizer, model)

            patient_id = extract_patient_id(file_path)

            row_data = {
                "patient_id": patient_id,
                "label": label
            }

            for i, val in enumerate(embedding):
                row_data[f"e_{i}"] = val

            dataset_rows.append(row_data)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSuccess! Extracted features for {len(df)} files.")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    extract_features()