"""
metrics.py

Evaluation metrics for Smart MCQ Solver.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


def recall(y_true, y_pred):
    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


def f1(y_true, y_pred):
    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )


def map3(actual, predicted):
    """
    actual : list[str]
    predicted : list[list[str]]
    """

    score = 0.0

    for truth, preds in zip(actual, predicted):

        for rank, pred in enumerate(preds[:3]):

            if pred == truth:
                score += 1 / (rank + 1)
                break

    return score / len(actual)
