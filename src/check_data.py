import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")

print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)
print("SAMPLE SHAPE:", sample.shape)

print("\nTRAIN COLUMNS")
print(train.columns.tolist())

print("\nTEST COLUMNS")
print(test.columns.tolist())

print("\nFIRST TRAIN ROW")
print(train.head(1))