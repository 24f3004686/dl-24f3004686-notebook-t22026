"""
preprocessing.py

Utility functions for loading and preprocessing
the Smart MCQ Solver dataset.
"""

from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path("data")


def load_train():
    """Load training dataset."""
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test():
    """Load test dataset."""
    return pd.read_csv(DATA_DIR / "test.csv")


def load_sample():
    """Load sample submission."""
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def clean_text(text):
    """
    Basic text cleaning.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataframe(df):

    df = df.copy()

    columns = ["prompt", "A", "B", "C", "D", "E"]

    for col in columns:
        df[col] = df[col].apply(clean_text)

    return df
