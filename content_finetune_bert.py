#https://thigm85.github.io/blog/search/cord19/bert/transformers/optuna/2020/11/07/bert-training-optuna-tuning.html
#https://github.com/MoritzLaurer/summer-school-transformers-2023/blob/main/3_tune_bert.ipynb
#https://www.kaggle.com/code/xylarwardhan/nlp-on-imbd-dataset-using-bert-and-optuna
#https://discuss.huggingface.co/t/multiple-training-will-give-exactly-the-same-result-except-for-the-first-time/8493
#https://machinelearningmastery.com/difference-test-validation-datasets/
#https://discuss.huggingface.co/t/datacollator-vs-tokenizers/5897
#https://github.com/google-research/bert
#https://medium.com/carbon-consulting/transformer-models-hyperparameter-optimization-with-the-optuna-299e185044a8
import pandas as pd
import numpy as np
import torch
import gc
import json
import optuna
import evaluate

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, set_seed, DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

CHECKPOINT = "distilbert-base-uncased"
CATEGORY = "V1"  # V11, V12, V2
TEXT_FIELD = "cleaned_shortcut"
SEED = 42
N_TRIALS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = f"{CATEGORY}_{CHECKPOINT}"
DATA_PATH = "round1_9_r1r2r3r4r5_corrected.csv"

set_seed(SEED)

df = pd.read_csv(DATA_PATH).sample(frac=1, random_state=SEED)
df = df[[CATEGORY, TEXT_FIELD]].dropna().drop_duplicates(subset=[TEXT_FIELD])
df.columns = ["labels", "text"]
df["labels"] = df["labels"].astype(int)
df["text"] = df["text"].astype(str)

label_list = sorted(df["labels"].unique())
label2id = {i: label for i, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label_list)

X_train, X_temp, y_train, y_temp = train_test_split(df["text"], df["labels"], test_size=0.2, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

hp_datasets = DatasetDict({
    "train": Dataset.from_dict({"text": X_train, "labels": y_train}),
    "validation": Dataset.from_dict({"text": X_val, "labels": y_val}),
    "test": Dataset.from_dict({"text": X_test, "labels": y_test})
})

fi_datasets = DatasetDict({
    "train": Dataset.from_dict({"text": pd.concat([X_train, X_val]), "labels": pd.concat([y_train, y_val])}),
    "test": Dataset.from_dict({"text": X_test, "labels": y_test})
})

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=True)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def tokenize_dataset(datasets):
    tokenized = datasets.map(tokenize_function, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized, collator

def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def model_init():
    clean_memory()
    config = AutoConfig.from_pretrained(CHECKPOINT, label2id=label2id, id2label=id2label, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, config=config)

metric = evaluate.combine(["f1", "accuracy", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def opt_train(datasets):
    tokenized_datasets, data_collator = tokenize_dataset(datasets)

    trainer = Trainer(
        model_init=model_init,
        args=TrainingArguments(
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            seed=SEED,
            output_dir=f"./results/{TRAIN_DIR}",
            logging_dir=f"./logs/{TRAIN_DIR}",
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    def my_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.1, 0.6, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        }

    def my_objective(metrics):
        pd.DataFrame({
            'f1': [metrics["eval_f1"]],
            'acc': [metrics["eval_accuracy"]],
            'recall': [metrics["eval_recall"]],
            'precision': [metrics["eval_precision"]]
        }).to_csv(f"{CATEGORY}_trial_metric_{CHECKPOINT}.csv", mode='a', header=False, index=False)
        return metrics["eval_f1"]

    pd.DataFrame(columns=["f1", "acc", "recall", "precision"]).to_csv(f"{CATEGORY}_trial_metric_{CHECKPOINT}.csv", index=False)

    optuna_sampler = optuna.samplers.TPESampler(
        seed=SEED,
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        n_startup_trials=N_TRIALS // 2,
        n_ei_candidates=24
    )

    best_run = trainer.hyperparameter_search(
        n_trials=N_TRIALS,
        compute_objective=my_objective,
        direction="maximize",
        hp_space=my_hp_space,
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna_sampler
    )

    return best_run

best_run = opt_train(hp_datasets)

with open(f"{CATEGORY}_bestrun_{CHECKPOINT}.json", "w") as f:
    json.dump(best_run.hyperparameters, f)

def final_train(datasets):
    tokenized_datasets, data_collator = tokenize_dataset(datasets)

    trainer = Trainer(
        model_init=model_init,
        args=TrainingArguments(
            output_dir=f"./results/{TRAIN_DIR}",
            seed=SEED,
            per_device_eval_batch_size=4,
            evaluation_strategy="no",
            save_strategy="no",
            report_to="all",
            **best_run.hyperparameters
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print(eval_metrics)
    trainer.save_model(f"{CATEGORY}_bestrun_{CHECKPOINT}")

final_train(fi_datasets)
