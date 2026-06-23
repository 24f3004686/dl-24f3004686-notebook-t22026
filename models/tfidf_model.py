"""
TF-IDF + Logistic Regression Model
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFModel:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=60000,
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.model = LogisticRegression(
            C=2.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        )
        
    def prepare_training_data(self, train_df):

        X_train = []
        y_train = []

        for _, row in train_df.iterrows():

            prompt = row["prompt"]

            for option in ["A", "B", "C", "D", "E"]:

                X_train.append(
                    prompt + " " + str(row[option])
                )

                y_train.append(
                    1 if row["answer"] == option else 0
                )

        return X_train, y_train
    
    def train(self, train_df):

        X_train, y_train = self.prepare_training_data(train_df)

        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.model.fit(X_train_vec, y_train)

        print("Training Complete!")
        
        
    def predict_top3(self, test_df):

        predictions = []

        for _, row in test_df.iterrows():

            scores = {}

            for option in ["A", "B", "C", "D", "E"]:

                text = row["prompt"] + " " + str(row[option])

                vec = self.vectorizer.transform([text])

                score = self.model.predict_proba(vec)[0][1]

                scores[option] = score

            ranked = sorted(
                scores,
                key=scores.get,
                reverse=True
            )

            predictions.append(ranked[:3])

        return predictions
    
    
    def evaluate(self, validation_df):

        predictions = self.predict_top3(validation_df)

        actual = validation_df["answer"].tolist()

        from src.metrics import map3

        score = map3(actual, predictions)

        print(f"MAP@3 : {score:.5f}")

        return score