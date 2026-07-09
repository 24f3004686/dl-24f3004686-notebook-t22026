import os
import numpy as np
import torch

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)


class ElectraEnsembleModel:

    def __init__(self):
        self.electra_name = "google/electra-base-discriminator"
        self.distil_name = "distilroberta-base"

        self.label2id = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4
        }

        self.id2label = {
            v: k
            for k, v in self.label2id.items()
        }

        # ELECTRA
        self.electra_tokenizer = AutoTokenizer.from_pretrained(
            self.electra_name
        )

        self.electra_model = AutoModelForSequenceClassification.from_pretrained(
            self.electra_name,
            num_labels=5,
            id2label=self.id2label,
            label2id=self.label2id
        )

        # DistilRoBERTa
        self.distil_tokenizer = AutoTokenizer.from_pretrained(
            self.distil_name
        )

        self.distil_model = AutoModelForSequenceClassification.from_pretrained(
            self.distil_name,
            num_labels=5,
            id2label=self.id2label,
            label2id=self.label2id
        )

        self.electra_collator = DataCollatorWithPadding(
            tokenizer=self.electra_tokenizer
        )

        self.distil_collator = DataCollatorWithPadding(
            tokenizer=self.distil_tokenizer
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

    def tokenize_electra(self, example):
        return self.electra_tokenizer(
            example["text"],
            truncation=True,
            max_length=384
        )

    def tokenize_distil(self, example):
        return self.distil_tokenizer(
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

        train_dataset = Dataset.from_pandas(train_split)
        valid_dataset = Dataset.from_pandas(valid_split)
        test_dataset = Dataset.from_pandas(test_df)

        train_dataset = train_dataset.map(self.prepare_text)
        valid_dataset = valid_dataset.map(self.prepare_text)
        test_dataset = test_dataset.map(self.prepare_text)

        # ELECTRA
        electra_train = train_dataset.map(
            self.tokenize_electra,
            batched=True
        )

        electra_valid = valid_dataset.map(
            self.tokenize_electra,
            batched=True
        )

        electra_test = test_dataset.map(
            self.tokenize_electra,
            batched=True
        )

        # DISTILROBERTA
        distil_train = train_dataset.map(
            self.tokenize_distil,
            batched=True
        )

        distil_valid = valid_dataset.map(
            self.tokenize_distil,
            batched=True
        )

        distil_test = test_dataset.map(
            self.tokenize_distil,
            batched=True
        )

        return (
            electra_train,
            electra_valid,
            electra_test,
            distil_train,
            distil_valid,
            distil_test
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

    def train_electra(self, train_dataset, valid_dataset):
        args = TrainingArguments(
            output_dir="./electra_results",
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
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=self.electra_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.electra_tokenizer,
            data_collator=self.electra_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        return trainer

    def train_distil(self, train_dataset, valid_dataset):
        args = TrainingArguments(
            output_dir="./distil_results",
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
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=self.distil_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            tokenizer=self.distil_tokenizer,
            data_collator=self.distil_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        return trainer

    def predict_electra(self, trainer, test_dataset):
        predictions = trainer.predict(
            test_dataset
        )

        return predictions.predictions

    def predict_distil(self, trainer, test_dataset):
        predictions = trainer.predict(
            test_dataset
        )

        return predictions.predictions
    
    
    def ensemble_predictions(self, electra_logits, distil_logits):
        ensemble_logits = (
            self.electra_weight * electra_logits +
            self.distil_weight * distil_logits
        )

        top3 = np.argsort(
            ensemble_logits,
            axis=1
        )[:, -3:][:, ::-1]

        predictions = [
            " ".join(
                self.id2label[idx]
                for idx in row
            )
            for row in top3
        ]

        return predictions
    
    
    def save_models(self, electra_trainer, distil_trainer, path="saved_models"):
        os.makedirs(
            path,
            exist_ok=True
        )

        electra_path = os.path.join(
            path,
            "electra"
        )

        distil_path = os.path.join(
            path,
            "distil"
        )

        electra_trainer.save_model(
            electra_path
        )

        self.electra_tokenizer.save_pretrained(
            electra_path
        )

        distil_trainer.save_model(
            distil_path
        )

        self.distil_tokenizer.save_pretrained(
            distil_path
        )

    print("Models saved successfully!")

    def load_models(self, path="saved_models"):
        electra_path = os.path.join(
            path,
            "electra"
        )

        distil_path = os.path.join(
            path,
            "distil"
        )

        self.electra_tokenizer = AutoTokenizer.from_pretrained(
            electra_path
        )

        self.electra_model = AutoModelForSequenceClassification.from_pretrained(
            electra_path
        )

        self.distil_tokenizer = AutoTokenizer.from_pretrained(
            distil_path
        )

        self.distil_model = AutoModelForSequenceClassification.from_pretrained(
            distil_path
        )

        print("Models loaded successfully!")


     

