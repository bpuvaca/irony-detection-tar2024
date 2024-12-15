import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

models = ["bert", "bertweet", "roberta"]
datasets = ["sarcasm", "polarity"]
mix_suffix = "_trained_on_mix_whole_4epoch.csv"
single_suffix = "_trainedon_{}_4epoch.csv"
folder_path = "lowest_correctness/all_predictions"
output_folder = "heatmaps"

os.makedirs(output_folder, exist_ok=True)

bins = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
labels = [0.0, 0.25, 0.5, 0.75, 1.0]

def prepare_correctness_bins(df):
    return pd.cut(df['mean_correctness'], bins=bins, labels=labels, include_lowest=True)

def match_data(single_task_df, mix_df):
    mix_df = mix_df.drop_duplicates(subset='tweet')
    merged_df = pd.merge(single_task_df, mix_df, on='tweet', suffixes=('_single', '_mix'))
    return merged_df

def create_heatmap(single_task_df, mix_df, model_name, dataset_name):
    matched_df = match_data(single_task_df, mix_df)

    single_task_bins = prepare_correctness_bins(matched_df[['mean_correctness_single']].rename(columns={'mean_correctness_single': 'mean_correctness'}))
    mix_bins = prepare_correctness_bins(matched_df[['mean_correctness_mix']].rename(columns={'mean_correctness_mix': 'mean_correctness'}))

    heatmap_data = pd.crosstab(index=mix_bins, columns=single_task_bins)

    all_labels = pd.Index(labels, name="Mix Correctness")
    heatmap_data = heatmap_data.reindex(index=all_labels[::-1], columns=all_labels, fill_value=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="RdYlGn", cbar_kws={'label': 'Count'})
    plt.title(f"Model: {model_name.upper()}, Dataset: {dataset_name.capitalize()}")
    plt.xlabel("Single Task Correctness")
    plt.ylabel("Mix Correctness")

    output_path = os.path.join(output_folder, f"{model_name}_{dataset_name}_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap: {output_path}")

for model in models:
    for dataset in datasets:
        single_task_file = os.path.join(folder_path, model + single_suffix.format(dataset))
        mix_file = os.path.join(folder_path, model + mix_suffix)

        single_task_df = pd.read_csv(single_task_file)
        mix_df = pd.read_csv(mix_file)

        if single_task_df is not None and mix_df is not None:
            create_heatmap(single_task_df, mix_df, model, dataset)
