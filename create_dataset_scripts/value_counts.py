import pandas as pd

print("MIX TRAIN")

df = pd.read_csv('../datasets/mix/mix_train.csv')

label_counts = df['label'].value_counts()

print(f"Number of 0s: {label_counts.get(0, 0)}")
print(f"Number of 1s: {label_counts.get(1, 0)}")

print("MIX VALID")

df = pd.read_csv('../datasets/mix/mix_valid.csv')

label_counts = df['label'].value_counts()

print(f"Number of 0s: {label_counts.get(0, 0)}")
print(f"Number of 1s: {label_counts.get(1, 0)}")

print("MIX TEST")

df = pd.read_csv('../datasets/mix/mix_test.csv')

label_counts = df['label'].value_counts()

print(f"Number of 0s: {label_counts.get(0, 0)}")
print(f"Number of 1s: {label_counts.get(1, 0)}")