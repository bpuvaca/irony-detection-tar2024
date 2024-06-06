import pandas as pd

sarcasm_df = pd.read_csv('irony-detection-tar2024/datasets/sarcasm/sarcasm_test.csv')
irony_df = pd.read_csv('irony-detection-tar2024/datasets/irony/irony_test.csv')

sarcasm_ones = sarcasm_df[sarcasm_df['label'] == 1]
irony_ones = irony_df[irony_df['label'] == 1]
irony_ones_sampled = irony_ones.sample(n=173, random_state=42)
all_ones = pd.concat([sarcasm_ones, irony_ones_sampled])

sarcasm_zeros = sarcasm_df[sarcasm_df['label'] == 0]
irony_zeros = irony_df[irony_df['label'] == 0]
all_zeros = pd.concat([sarcasm_zeros, irony_zeros])
all_zeros_sampled = all_zeros.sample(n=346, random_state=42)


mixed_df = pd.concat([all_ones, all_zeros_sampled])
mixed_df = mixed_df.sample(frac=1, random_state=42).reset_index(drop=True)
mixed_df.to_csv('mix_test.csv', index=False)

print("New dataset 'mix_test.csv' created successfully.")
