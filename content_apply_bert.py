import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

CATEGORY = "V1"
CHECKPOINT = "distilbert-base-uncased"
MODEL_DIR = f"{CATEGORY}_bestrun_{CHECKPOINT}"
TEXT_COLUMN = "cleaned_shortcut"
SEED_GLOBAL = 42

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True, model_max_length=512)

classifier = pipeline(
    "text-classification",
    model=f"./{MODEL_DIR}",
    tokenizer=tokenizer,
    truncation=True,
    framework="pt",
    device=device,
)

df = pd.read_pickle("revids_comments.pkl")
print(f"Total revids: {len(df)}")

groundtruth = pd.read_csv("round1_9_r1r2r3r4r5_corrected.csv")
known_revids = groundtruth.dropna(subset=["revid"])["revid"].tolist()

df = df[~df["revid"].isin(known_revids)]
print(f"Unseen revids: {len(df)}")

df = df[["revid", TEXT_COLUMN]].dropna(subset=[TEXT_COLUMN]).copy()
df.columns = ["revid", "text"]

text_list = df["text"].tolist()

pipe_output = classifier(
    text_list,
    batch_size=4  # Reduce if running out of memory
)

df_output = pd.DataFrame(pipe_output)
df["label_pred"] = df_output["label"].tolist()
df["label_pred_probability"] = df_output["score"].round(2).tolist()

df.to_pickle(f"{CATEGORY}_revids_comments_classified.pkl")
