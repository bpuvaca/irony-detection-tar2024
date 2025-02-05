import pandas as pd 
import sys
sys.path.append('C:/Users/Florijan/Documents/FER/Diplomski/2.Semestar/TAR/projekt/irony-detection-tar2024/nemojte')
from liwc_analysis import get_most_important_features, get_features

irony_liwc = pd.read_csv('../results/LIWC/datasets/liwc_irony_parsed.csv', sep=',')
sarcasm_liwc = pd.read_csv('../results/LIWC/datasets/liwc_sarcasm_parsed.csv', sep=',')

irony_features = get_most_important_features('../results/LIWC/datasets/liwc_irony_parsed.csv')
sarcasm_features = get_most_important_features('../results/LIWC/datasets/liwc_sarcasm_parsed.csv')

irony_liwc_filtered = irony_liwc[['label', 'tweet'] + [feature for feature in irony_features.keys()]]
sarcasm_liwc_filtered = sarcasm_liwc[['label', 'tweet'] + [feature for feature in sarcasm_features.keys()]]

irony_liwc_filtered.to_csv('../datasets/LIWC/liwc_irony_parsed_filtered.csv', index=False)
sarcasm_liwc_filtered.to_csv('../datasets/LIWC/liwc_sarcasm_parsed_filtered.csv', index=False)