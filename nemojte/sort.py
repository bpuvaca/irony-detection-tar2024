import pandas as pd

df = pd.read_csv("lowest_correctness/all_predictions/roberta_trained_on_semeval_mix_4epoch.csv")
df_sorted = df.sort_values(by="mean_correctness", ascending=True)
df_sorted.to_csv("lowest_correctness/all_predictions/roberta_trained_on_semeval_mix_4epoch.csv", index=False)
