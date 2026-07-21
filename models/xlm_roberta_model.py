"""
XLM-RoBERTa Base Model
"""

import os
import numpy as np
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

class XLMRobertaModel:

    def __init__(self):

        self.model_name = "xlm-roberta-base"

        self.label2id = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
        }

        self.id2label = {
            v: k for k, v in self.label2id.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=5,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def prepare_text(self, example):
        return {
            "text": (
                f"Question:\n{example['prompt']}\n\n"
                f"Options:\n"
                f"A. {example['A']}\n"
                f"B. {example['B']}\n"
                f"C. {example['C']}\n"
                f"D. {example['D']}\n"
                f"E. {example['E']}\n\n"
                "Choose the correct answer."
            )
        }
        
        
    def tokenize(self, example):
        return self.tokenizer(
            example["text"],
            truncation=True,
            max_length=384
        )
    
    def prepare_datasets(self, train_df, test_df):
        train_df = train_df.copy()

        train_df["label"] = train_df["answer"].map(
            self.label2id
        )

        train_split, valid_split = train_test_split(
            train_df,
            test_size=0.10,
            stratify=train_df["label"],
            random_state=42
        )

        train_dataset = Dataset.from_pandas(
            train_split
        )

        valid_dataset = Dataset.from_pandas(
            valid_split
        )

        test_dataset = Dataset.from_pandas(
            test_df
        )

        train_dataset = train_dataset.map(
            self.prepare_text
        )

        valid_dataset = valid_dataset.map(
            self.prepare_text
        )

        test_dataset = test_dataset.map(
            self.prepare_text
        )

        train_dataset = train_dataset.map(
            self.tokenize,
            batched=True
        )

        valid_dataset = valid_dataset.map(
            self.tokenize,
            batched=True
        )

        test_dataset = test_dataset.map(
            self.tokenize,
            batched=True
        )

        return (
            train_dataset,
            valid_dataset,
            test_dataset,
        )
        
        
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted"
        )

        accuracy = accuracy_score(
            labels,
            predictions
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self, train_df, test_df):
        (
            train_dataset,
            valid_dataset,
            self.test_dataset
        ) = self.prepare_datasets(
            train_df,
            test_df
        )

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=50,
            save_total_limit=1,
            fp16=torch.cuda.is_available()
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

        print("Training Complete!")
        
    def predict_top3(self):
        predictions = self.trainer.predict(
            self.test_dataset
        )

        logits = predictions.predictions

        top3 = np.argsort(
            logits,
            axis=1
        )[:, -3:][:, ::-1]

        final_predictions = []

        for row in top3:
            final_predictions.append(
                " ".join(
                    self.id2label[idx]
                    for idx in row
                )
            )

        return final_predictions

    def save_model(self, path="saved_models/xlm_roberta"):
        os.makedirs(
            path,
            exist_ok=True
        )

        self.trainer.save_model(path)

        self.tokenizer.save_pretrained(path)

        print(f"Model saved to {path}")

    def load_model(self, path="saved_models/xlm_roberta"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            path
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            path
        )

        print(f"Model loaded from {path}")


