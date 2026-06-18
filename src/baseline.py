import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = []
y_train = []

# Convert each MCQ into 5 training samples
for _, row in train.iterrows():
    prompt = row["prompt"]

    for option in ["A", "B", "C", "D", "E"]:
        text = prompt + " " + str(row[option])

        X_train.append(text)

        if row["answer"] == option:
            y_train.append(1)
        else:
            y_train.append(0)

print("Training samples:", len(X_train))

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)

# Classifier
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

print("Model trained!")

# Predict test set
predictions = []

for _, row in test.iterrows():

    scores = {}

    for option in ["A", "B", "C", "D", "E"]:

        text = row["prompt"] + " " + str(row[option])

        vec = vectorizer.transform([text])

        score = model.predict_proba(vec)[0][1]

        scores[option] = score

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    predictions.append(
        " ".join(ranked[:3])
    )

submission = pd.DataFrame({
    "ID": test["id"],
    "Prediction": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv created!")