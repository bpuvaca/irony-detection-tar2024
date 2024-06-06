import pandas as pd
from sklearn.model_selection import train_test_split


irony_df = pd.read_csv('irony.csv')

cleaned_irony_df = irony_df[irony_df['label'] != 3]

train_df, temp_df = train_test_split(cleaned_irony_df, train_size=0.7, stratify=cleaned_irony_df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, train_size=0.5, stratify=temp_df['label'], random_state=42)

train_df.to_csv('irony_train.csv', index=False)
val_df.to_csv('irony_validation.csv', index=False)
test_df.to_csv('irony_test.csv', index=False)

print("Train set label distribution:")
print(train_df['label'].value_counts())
print("Validation set label distribution:")
print(val_df['label'].value_counts())
print("Test set label distribution:")
print(test_df['label'].value_counts())

print("Train, validation, and test datasets saved to 'irony_train.csv', 'irony_validation.csv', and 'irony_test.csv'")