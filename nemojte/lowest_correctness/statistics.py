import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

models = ["bert", "bertweet", "roberta"]
datasets = ["polarity", "sarcasm"]

results = []

for model in models:
    for dataset in datasets:
        combined_file = f"{model}_{dataset}_combined.csv"

        combined_df = pd.read_csv(combined_file)

        mix_mean = combined_df['mean_correctness_mix'].mean()
        single_task_mean = combined_df[f'mean_correctness_{dataset}'].mean()

        mix_median = combined_df['mean_correctness_mix'].median()
        single_task_median = combined_df[f'mean_correctness_{dataset}'].median()

        mix_std = combined_df['mean_correctness_mix'].std()
        single_task_std = combined_df[f'mean_correctness_{dataset}'].std()

        t_stat, p_value = ttest_rel(combined_df[f'mean_correctness_{dataset}'], combined_df['mean_correctness_mix'])

        results.append({
            "Model": model,
            "Dataset": dataset,
            "Single task Mean": single_task_mean,
            "Mix Mean": mix_mean,
            "Single task Median": single_task_median,
            "Mix Median": mix_median,
            "Single task Std Dev": single_task_std,
            "Mix Std Dev": mix_std,
            "T-Stat": t_stat,
            "P-Value": p_value
        })

results_df = pd.DataFrame(results)

print(results_df)
