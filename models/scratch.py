# ==========================================================
# Scratch BiLSTM MCQ Solver
# Smart MCQ Solver Challenge
# ==========================================================

import os
import re
import random
import string
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

# ==========================================================
# Configuration
# ==========================================================

SEED = 42

MAX_LENGTH = 128

EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
}

LABEL2ID = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Data Loading
# ==========================================================

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train Shape : {train_df.shape}")
    print(f"Test Shape  : {test_df.shape}")

    return train_df, test_df

def clean_text(text):
    """
    Clean and normalize input text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted characters
    text = re.sub(r"[^a-z0-9.,!?;:()'\"%- ]", "", text)

    return text.strip()

def combine_text(row):
    """
    Combine question and answer choices into a single sequence.
    """

    return (
        f"Question: {clean_text(row['prompt'])} "
        f"Option A: {clean_text(row['A'])} "
        f"Option B: {clean_text(row['B'])} "
        f"Option C: {clean_text(row['C'])} "
        f"Option D: {clean_text(row['D'])} "
        f"Option E: {clean_text(row['E'])}"
    )
    
def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from training texts.
    """

    counter = Counter()

    for text in texts:
        counter.update(text.split())

    vocab = dict(SPECIAL_TOKENS)

    for word, freq in counter.items():
        if freq >= min_freq:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab

def encode_text(text, vocab):
    """
    Convert words into token IDs.
    """

    return [
        vocab.get(word, vocab["<UNK>"])
        for word in text.split()
    ]
    
def pad_sequence(sequence, max_length=MAX_LENGTH):
    """
    Pad or truncate sequences.
    """

    if len(sequence) >= max_length:
        return sequence[:max_length]

    return sequence + [vocab["<PAD>"]] * (max_length - len(sequence))


def preprocess_dataframe(df):
    """
    Apply preprocessing to an entire dataframe.
    """

    processed = df.copy()

    processed["text"] = processed.apply(combine_text, axis=1)

    return processed


train_df = preprocess_dataframe(train_df)
test_df = preprocess_dataframe(test_df)

vocab = build_vocab(train_df["text"])

print(f"Vocabulary Size: {len(vocab):,}")

train_df["input_ids"] = train_df["text"].apply(
    lambda x: pad_sequence(
        encode_text(x, vocab)
    )
)

test_df["input_ids"] = test_df["text"].apply(
    lambda x: pad_sequence(
        encode_text(x, vocab)
    )
)


# ==========================================================
# Train Validation Split
# ==========================================================

def create_splits(train_df):
    """
    Split the training data into train and validation sets.
    """

    train_split, valid_split = train_test_split(
        train_df,
        test_size=0.1,
        random_state=SEED,
        stratify=train_df["answer"]
    )

    train_split = train_split.reset_index(drop=True)
    valid_split = valid_split.reset_index(drop=True)

    return train_split, valid_split


# ==========================================================
# PyTorch Dataset
# ==========================================================

class MCQDataset(Dataset):
    """
    Dataset for Multiple Choice Question Answering.
    """

    def __init__(self, dataframe, is_test=False):

        self.df = dataframe
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        input_ids = torch.tensor(
            row["input_ids"],
            dtype=torch.long
        )

        if self.is_test:

            return {
                "input_ids": input_ids
            }

        label = torch.tensor(
            LABEL2ID[row["answer"]],
            dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": label
        }
        
# ==========================================================
# DataLoaders
# ==========================================================

def create_dataloaders(
    train_df,
    valid_df,
    test_df
):

    train_dataset = MCQDataset(train_df)

    valid_dataset = MCQDataset(valid_df)

    test_dataset = MCQDataset(
        test_df,
        is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )
    
    
# ==========================================================
# Verify DataLoader
# ==========================================================

def inspect_batch(loader):

    batch = next(iter(loader))

    print("="*50)

    print("Input Shape :", batch["input_ids"].shape)

    if "labels" in batch:
        print("Label Shape :", batch["labels"].shape)

    print("="*50)
    
# ==========================================================
# Attention Layer
# ==========================================================

class Attention(nn.Module):
    """
    Simple attention mechanism over BiLSTM outputs.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):

        # lstm_output:
        # (batch_size, seq_len, hidden_dim*2)

        scores = self.attention(lstm_output)

        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * lstm_output, dim=1)

        return context
    
# ==========================================================
# Scratch BiLSTM Model
# ==========================================================

class ScratchMCQModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=SPECIAL_TOKENS["<PAD>"]
        )

        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )

        self.attention = Attention(HIDDEN_DIM)

        self.dropout = nn.Dropout(DROPOUT)

        self.classifier = nn.Linear(
            HIDDEN_DIM * 2,
            len(LABEL2ID)
        )

    def forward(self, input_ids):

        embeddings = self.embedding(input_ids)

        lstm_output, _ = self.lstm(embeddings)

        context = self.attention(lstm_output)

        context = self.dropout(context)

        logits = self.classifier(context)

        return logits
    
    
# ==========================================================
# Weight Initialization
# ==========================================================

def initialize_weights(model):

    for name, param in model.named_parameters():

        if "weight" in name:

            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        elif "bias" in name:

            nn.init.constant_(param, 0)
            
            
# ==========================================================
# Build Model
# ==========================================================

def build_model(vocab_size):

    model = ScratchMCQModel(vocab_size)

    initialize_weights(model)

    model = model.to(DEVICE)

    return model


# ==========================================================
# Model Summary
# ==========================================================

def model_summary(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 50)
    print(model)
    print("=" * 50)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print("=" * 50)
    
    
# ==========================================================
# Forward Pass Test
# ==========================================================

def test_forward_pass(model, loader):

    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(DEVICE)

    outputs = model(input_ids)

    print("=" * 50)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", outputs.shape)
    print("=" * 50)
    
    
# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(model, dataloader, optimizer, criterion):

    model.train()

    total_loss = 0
    predictions = []
    labels = []

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)
        target = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids)

        loss = criterion(outputs, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        labels.extend(target.cpu().numpy())

        progress_bar.set_postfix(
            loss=loss.item()
        )

    accuracy = accuracy_score(labels, predictions)

    return total_loss / len(dataloader), accuracy


# ==========================================================
# Validation
# ==========================================================

def validate(model, dataloader, criterion):

    model.eval()

    total_loss = 0

    predictions = []
    labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            target = batch["labels"].to(DEVICE)

            outputs = model(input_ids)

            loss = criterion(outputs, target)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)

    return total_loss / len(dataloader), accuracy



# ==========================================================
# Model Training
# ==========================================================

def train_model(model, train_loader, valid_loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-"*50)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        valid_loss, valid_acc = validate(
            model,
            valid_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")

        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Valid Acc  : {valid_acc:.4f}")

        if valid_acc > best_accuracy:

            best_accuracy = valid_acc

            torch.save(
                model.state_dict(),
                "best_scratch_model.pth"
            )

            print("✅ Best model saved!")

    print("\nTraining Complete")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    return history


# ==========================================================
# Training Curves
# ==========================================================

def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10,5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.show()

    plt.figure(figsize=(10,5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["valid_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")

    plt.legend()

    plt.show()
    
    
# ==========================================================
# Load Best Model
# ==========================================================

def load_best_model(model):

    model.load_state_dict(
        torch.load(
            "best_scratch_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model


# ==========================================================
# Prediction
# ==========================================================

def predict(model, test_loader):

    model.eval()

    predictions = []

    with torch.no_grad():

        for batch in tqdm(test_loader, desc="Predicting"):

            input_ids = batch["input_ids"].to(DEVICE)

            outputs = model(input_ids)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())

    return predictions


# ==========================================================
# Convert Predictions
# ==========================================================

def decode_predictions(predictions):

    return [ID2LABEL[pred] for pred in predictions]


# ==========================================================
# Save Submission
# ==========================================================

def save_submission(test_df, predictions):

    submission = pd.DataFrame({
        "id": test_df["id"],
        "answer": predictions
    })

    submission.to_csv(
        "submission.csv",
        index=False
    )

    print("=" * 50)
    print("Submission saved successfully!")
    print(submission.head())
    print("=" * 50)
    
    
    
# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("Scratch BiLSTM MCQ Solver")
    print("=" * 60)

    # ------------------------------------------------------
    # Set Seed
    # ------------------------------------------------------

    set_seed()

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    train_df, test_df = load_data(
        "train.csv",
        "test.csv"
    )

    # ------------------------------------------------------
    # Text Preprocessing
    # ------------------------------------------------------

    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # ------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------

    vocab = build_vocab(train_df["text"])

    print(f"Vocabulary Size : {len(vocab):,}")

    # ------------------------------------------------------
    # Encoding
    # ------------------------------------------------------

    train_df["input_ids"] = train_df["text"].apply(
        lambda x: pad_sequence(
            encode_text(x, vocab)
        )
    )

    test_df["input_ids"] = test_df["text"].apply(
        lambda x: pad_sequence(
            encode_text(x, vocab)
        )
    )

    # ------------------------------------------------------
    # Train Validation Split
    # ------------------------------------------------------

    train_split, valid_split = create_splits(train_df)

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_split,
        valid_split,
        test_df
    )

    inspect_batch(train_loader)

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = build_model(len(vocab))

    model_summary(model)

    test_forward_pass(model, train_loader)

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    history = train_model(
        model,
        train_loader,
        valid_loader
    )

    # ------------------------------------------------------
    # Plot Curves
    # ------------------------------------------------------

    plot_history(history)

    # ------------------------------------------------------
    # Best Model
    # ------------------------------------------------------

    model = load_best_model(model)

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    predictions = predict(
        model,
        test_loader
    )

    predictions = decode_predictions(predictions)

    # ------------------------------------------------------
    # Submission
    # ------------------------------------------------------

    save_submission(
        test_df,
        predictions
    )

    print("Pipeline Finished Successfully!")




