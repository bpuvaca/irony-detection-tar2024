import pandas as pd

models = ["bert", "bertweet", "roberta"]
datasets = ["polarity", "sarcasm"]
mix_suffix = "trained_on_mix_whole_4epoch.csv"

for model in models:
    mix_file = f"{model}_{mix_suffix}"
    mix_df = pd.read_csv(mix_file)

    for dataset in datasets:
        specific_file = f"{model}_trainedon_{dataset}_4epoch.csv"
        specific_df = pd.read_csv(specific_file)

        merged_df = specific_df.merge(mix_df, on="tweet", suffixes=(f"_{dataset}", "_mix"))
        output_file = f"{model}_{dataset}_combined.csv"
        merged_df.to_csv(output_file, index=False, encoding="utf-8")
