"""
Word2Vec + Logistic Regression Model
"""

from email.mime import text

import numpy as np
import nltk

from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression

nltk.download("punkt", quiet=True)

class Word2VecModel:

    def __init__(self):

        self.embedding_size = 200

        self.w2v = None

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

                text = prompt + " " + str(row[option])

                X_train.append(text)

                y_train.append(
                    1 if row["answer"] == option else 0
                )

        return X_train, y_train
    
    def train_word2vec(self, texts):

        tokenized = [
            nltk.word_tokenize(text.lower())
            for text in texts
        ]

        self.w2v = Word2Vec(
            sentences=tokenized,
            vector_size=self.embedding_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1,
            epochs=15
        )
     
    def sentence_vector(self, text):

        words = text.lower().split()

        vectors = []

        for word in words:

            if word in self.w2v.wv:

                vectors.append(
                    self.w2v.wv[word]
                )

        if len(vectors) == 0:

            return np.zeros(self.embedding_size)

        return np.mean(vectors, axis=0)   
    
    def train(self, train_df):
        
        X_train, y_train = self.prepare_training_data(train_df)

        self.train_word2vec(X_train)

        X_train_vec = np.array([
            self.sentence_vector(text)
            for text in X_train
        ])

        self.model.fit(X_train_vec, y_train)

        print("Training Complete!")

        
    def predict_top3(self, test_df):

        predictions = []

        for _, row in test_df.iterrows():

            scores = {}

            for option in ["A", "B", "C", "D", "E"]:

                text = row["prompt"] + " " + str(row[option])

                vec = self.sentence_vector(text).reshape(1, -1)

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