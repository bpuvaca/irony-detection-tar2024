from transformer_main import train_and_evaluate, evaluate_only
import evaluate
import pandas as pd

def extract_preds(df, trained_on, tested_on):
    
    fp = df[(df['label'] == 0) & (df['prediction'] == 1)]
    fn = df[(df['label'] == 1) & (df['prediction'] == 0)]
    tp = df[(df['label'] == 1) & (df['prediction'] == 1)]
    tn = df[(df['label'] == 0) & (df['prediction'] == 0)]
    print(f'test_examples: {len(df)}\ntp count: {len(tp)}\nfp count: {len(fp)}\nfn count: {len(fn)}\ntn count: {len(tn)}')
    fp.to_csv(f'isarcasm_label_analysis/train_{trained_on}_test_{tested_on}_fp.csv', index=False)
    fn.to_csv(f'isarcasm_label_analysis/train_{trained_on}_test_{tested_on}_fn.csv', index=False)



datasets = ['isarcasm_sarc']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

# for model in models:
#     for dataset in datasets:
#         train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{params_suffix}", None, False, True)

# for model in models:
#     evaluate_only(model, f"{model}_isarcasm_sarc_", "isarcasm_irony", False, True)

# evaluate_only('bertweet', f"bertweet_isarcasm_sarc_", "isarcasm_sarc", False, True)
# file_path = 'all_preds/bertweet_isarcasm_sarc__test_on_isarcasm_sarc.csv'
# df = pd.read_csv(file_path)
# extract_preds(df, "sarcasm", "sarcasm")


# evaluate_only('bertweet', f"bertweet_isarcasm_sarc_", "isarcasm_irony", False, True)
# file_path = 'all_preds/bertweet_isarcasm_sarc__test_on_isarcasm_irony.csv'
# df = pd.read_csv(file_path)
# extract_preds(df, "sarcasm", "irony")


# evaluate_only('bertweet', 'bertweet_isarcasm_sarc_', 'semeval_polarity', False, True)
# file_path = 'all_preds/bertweet_isarcasm_sarc__test_on_semeval_polarity.csv'
# df = pd.read_csv(file_path)
# extract_preds(df, "sarcasm", "polarity")

# evaluate_only('bertweet', 'bertweet_isarcasm_sarc_', 'semeval_other', False, True)
# file_path = 'all_preds/bertweet_isarcasm_sarc__test_on_semeval_other.csv'
# df = pd.read_csv(file_path)
# extract_preds(df, "sarcasm", "other")

train_and_evaluate("semeval_polarity", "bertweet", None, f"bertweet_semeval_polarity_{params_suffix}", None, False, True)
file_path = 'all_preds/bertweet+semeval_polarity_test_on_semeval_polarity.csv'
df = pd.read_csv(file_path)
extract_preds(df, "polarity", "polarity")


evaluate_only('bertweet', f"bertweet_semeval_polarity_", "isarcasm_irony", False, True)
file_path = 'all_preds/bertweet_semeval_polarity__test_on_isarcasm_irony.csv'
df = pd.read_csv(file_path)
extract_preds(df, "polarity", "irony")


evaluate_only('bertweet', 'bertweet_semeval_polarity_', 'isarcasm_sarc', False, True)
file_path = 'all_preds/bertweet_semeval_polarity__test_on_isarcasm_sarc.csv'
df = pd.read_csv(file_path)
extract_preds(df, "polarity", "sarcasm")

evaluate_only('bertweet', 'bertweet_semeval_polarity_', 'semeval_other', False, True)
file_path = 'all_preds/bertweet_semeval_polarity__test_on_semeval_other.csv'
df = pd.read_csv(file_path)
extract_preds(df, "polarity", "other")