# dl-24f3004686-notebook-t22026

# 🧠 Smart MCQ Solver

An end-to-end Natural Language Processing project for solving Multiple Choice Questions using classical Machine Learning, Neural Networks, and Transformer-based Language Models.

The project was developed progressively across multiple milestones, starting with a TF-IDF + Logistic Regression baseline and progressing to pretrained Transformer architectures, retrieval-based experiments, and a Deep Learning model implemented from scratch.

---

## 📌 Problem Statement

The objective of the Smart MCQ Solver is to predict the correct answer for a multiple-choice question containing five candidate options:

- A
- B
- C
- D
- E

Unlike conventional classification problems where only the top prediction matters, this task evaluates the ranking of the candidate answers using **Mean Average Precision at 3 (MAP@3)**.

The system therefore produces a ranked list of the top three candidate answers for every question.

---

## 🎯 Objectives

- Perform exploratory data analysis and understand the MCQ dataset.
- Build a strong classical Machine Learning baseline.
- Experiment with neural network architectures.
- Fine-tune multiple pretrained Transformer models.
- Compare models using MAP@3 and standard classification metrics.
- Explore retrieval-based approaches for contextual augmentation.
- Implement a Deep Learning model from scratch.
- Maintain reproducible experiments using Git and Weights & Biases.
- Generate final ranked predictions for Kaggle submission.

---

## 🗂️ Project Evolution

The project was developed progressively:

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF + Logistic Regression
   │
   ▼
Feed Forward Neural Network
   │
   ▼
Transformer Models
(BERT, SciBERT, RoBERTa, DistilRoBERTa,
XLM-RoBERTa, ConvBERT, ELECTRA)
   │
   ▼
Retrieval / RAG Experiments
   │
   ▼
Scratch Deep Learning Model
(BiLSTM + Attention)
   │
   ▼
Evaluation & Kaggle Submission
