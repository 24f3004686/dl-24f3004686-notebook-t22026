"""
BERT Base Model
Smart MCQ Solver Challenge
"""

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


class BertMCQModel:
    
    def __init__(self):

        self.model_name = "bert-base-uncased"

        self.label_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4
        }

        self.inverse_label_map = {
            v: k for k, v in self.label_map.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=5
        )
        
    def prepare_text(self, df):

        df = df.copy()

        df["text"] = (
            df["prompt"].fillna("")
            + " [SEP] A: " + df["A"].fillna("")
            + " [SEP] B: " + df["B"].fillna("")
            + " [SEP] C: " + df["C"].fillna("")
            + " [SEP] D: " + df["D"].fillna("")
            + " [SEP] E: " + df["E"].fillna("")
        )

        return df
    
    def tokenize(self, examples):

        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
    @staticmethod
    def compute_metrics(eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )

        accuracy = (
            predictions == labels
        ).mean()

        return {
            "accuracy": accuracy
        }
        
    def train(self, train_df):

        train_df = self.prepare_text(train_df)

        train_df["label"] = train_df["answer"].map(
            self.label_map
        )

        train_data, valid_data = train_test_split(
            train_df,
            test_size=0.1,
            random_state=42,
            stratify=train_df["label"]
        )

        train_dataset = Dataset.from_pandas(
            train_data[["text", "label"]]
        )

        valid_dataset = Dataset.from_pandas(
            valid_data[["text", "label"]]
        )

        train_dataset = train_dataset.map(
            self.tokenize,
            batched=True
        )

        valid_dataset = valid_dataset.map(
            self.tokenize,
            batched=True
        )

        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label"
            ]
        )

        valid_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "label"
            ]
        )

        training_args = TrainingArguments(

            output_dir="./bert_mcq",

            eval_strategy="epoch",

            save_strategy="epoch",

            learning_rate=2e-5,

            per_device_train_batch_size=8,

            per_device_eval_batch_size=8,

            num_train_epochs=3,

            warmup_steps=100,

            weight_decay=0.01,

            load_best_model_at_end=True,

            report_to="none"
        )

        trainer = Trainer(

            model=self.model,

            args=training_args,

            train_dataset=train_dataset,

            eval_dataset=valid_dataset,

            processing_class=self.tokenizer,

            compute_metrics=self.compute_metrics
        )

        trainer.train()

        return trainer
    
    def predict(self, trainer, test_df):

        test_df = self.prepare_text(test_df)

        test_dataset = Dataset.from_pandas(
            test_df[["text"]]
        )

        test_dataset = test_dataset.map(
            self.tokenize,
            batched=True
        )

        test_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask"
            ]
        )

        logits = trainer.predict(
            test_dataset
        ).predictions

        probs = torch.softmax(
            torch.tensor(logits),
            dim=1
        ).numpy()

        labels = np.array([
            "A",
            "B",
            "C",
            "D",
            "E"
        ])

        top3 = np.argsort(
            probs,
            axis=1
        )[:, -3:][:, ::-1]

        predictions = [
            " ".join(labels[idx])
            for idx in top3
        ]

        return predictions
    
    
    

