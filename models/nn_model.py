import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

class MCQNet(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512,128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,2)

        )

    def forward(self,x):

        return self.network(x)      
        
class NeuralModel:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=60000,
            ngram_range=(1,3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

        self.model = None
        
    def prepare_training_data(self, train_df):

        X_train = []
        y_train = []

        for _, row in train_df.iterrows():

            prompt = row["prompt"]

            for option in ["A","B","C","D","E"]:

                X_train.append(
                    prompt + " " + str(row[option])
                )

                y_train.append(
                    1 if row["answer"] == option else 0
             )

        return X_train, y_train
    
    def train(self, train_df, epochs=5):

        X_train, y_train = self.prepare_training_data(train_df)

        X_train_vec = self.vectorizer.fit_transform(X_train)

        X_train_tensor = torch.FloatTensor(
            X_train_vec.toarray()
        )

        y_train_tensor = torch.LongTensor(y_train)

        input_size = X_train_tensor.shape[1]

        self.model = MCQNet(input_size)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        for epoch in range(epochs):

            optimizer.zero_grad()

            outputs = self.model(X_train_tensor)

            loss = criterion(
                outputs,
                y_train_tensor
            )

            loss.backward()

            optimizer.step()

            print(
                f"Epoch {epoch+1}/{epochs}  Loss: {loss.item():.4f}"
            )

        print("Training Complete!")
    
    def predict_top3(self, test_df):

        predictions = []

        self.model.eval()

        with torch.no_grad():

            for _, row in test_df.iterrows():

                scores = {}

                for option in ["A", "B", "C", "D", "E"]:

                    text = row["prompt"] + " " + str(row[option])

                    vec = self.vectorizer.transform([text])

                    vec = torch.FloatTensor(vec.toarray())

                    output = self.model(vec)

                    probability = torch.softmax(output, dim=1)[0][1].item()

                    scores[option] = probability

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


