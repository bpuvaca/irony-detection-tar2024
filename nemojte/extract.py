import pandas as pd
import os

models = ['bert', 'bertweet', 'roberta']
base_dir = 'lowest_correctness'

# Process each model
for model in models:
    mix_file = pd.read_csv(os.path.join(base_dir, f'{model}_trainedon_mix_4epoch.csv'))
    polarity_file = pd.read_csv(os.path.join(base_dir, f'{model}_trainedon_polarity_4epoch.csv'))
    sarcasm_file = pd.read_csv(os.path.join(base_dir, f'{model}_trainedon_sarcasm_4epoch.csv'))

    result_rows = []

    for _, row in polarity_file.iterrows():
        if row['tweet'] in mix_file['tweet'].values:
            mix_row = mix_file[mix_file['tweet'] == row['tweet']].iloc[0]
            result_rows.append({
                "tweet": row['tweet'],
                "label": row['label'],
                "mean_confidence_mix": mix_row['mean_confidence'],
                "variability_mix": mix_row['variability'],
                "mean_correctness_mix": mix_row['mean_correctness'],
                "p_s": 0,
                "mean_confidence_x": row['mean_confidence'],
                "variability_x": row['variability'],
                "mean_correctness_x": row['mean_correctness']
            })

    for _, row in sarcasm_file.iterrows():
        if row['tweet'] in mix_file['tweet'].values:
            mix_row = mix_file[mix_file['tweet'] == row['tweet']].iloc[0]
            result_rows.append({
                "tweet": row['tweet'],
                "label": row['label'],
                "mean_confidence_mix": mix_row['mean_confidence'],
                "variability_mix": mix_row['variability'],
                "mean_correctness_mix": mix_row['mean_correctness'],
                "p_s": 1,
                "mean_confidence_x": row['mean_confidence'],
                "variability_x": row['variability'],
                "mean_correctness_x": row['mean_correctness']
            })

    columns = [
        "tweet", "label", "mean_confidence_mix", "variability_mix",
        "mean_correctness_mix", "p_s", "mean_confidence_x",
        "variability_x", "mean_correctness_x"
    ]
    final_df = pd.DataFrame(result_rows, columns=columns)

    output_file_path = os.path.join(base_dir, f'{model}_overlap_with_metrics.csv')
    final_df.to_csv(output_file_path, index=False)

    print(f"File saved for {model}: {output_file_path}")
