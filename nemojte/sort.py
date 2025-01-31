import pandas as pd

df = pd.read_csv("lowest_correctness/bf_predictions/roberta_trained_on_bf_irony_roberta_4epoch.csv")
df_sorted = df.sort_values(by="mean_correctness", ascending=True)
df_sorted.to_csv("lowest_correctness/bf_predictions/roberta_trained_on_bf_irony_roberta_4epoch.csv", index=False)
