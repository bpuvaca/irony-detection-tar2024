import pandas as pd

def extract_preds(df, trained_on, tested_on):
    
    fp = df[(df['label'] == 0) & (df['prediction'] == 1)]
    fn = df[(df['label'] == 1) & (df['prediction'] == 0)]
    tp = df[(df['label'] == 1) & (df['prediction'] == 1)]
    tn = df[(df['label'] == 0) & (df['prediction'] == 0)]
    print(f'test_examples: {len(df)}\ntp count: {len(tp)}\nfp count: {len(fp)}\nfn count: {len(fn)}\ntn count: {len(tn)}')
    fp.to_csv(f'train_{trained_on}_test_{tested_on}_fp.csv', index=False)
    fn.to_csv(f'train_{trained_on}_test_{tested_on}_fn.csv', index=False)


# Load the dataframe
file_path = '../all_preds/bertweet+isarcasm_sarc_test_on_isarcasm_sarc.csv'
df = pd.read_csv(file_path)

extract_preds(df, "sarcasm", "sarcasm")

file_path = '../all_preds/bertweet+isarcasm_irony_test_on_isarcasm_irony.csv'
df = pd.read_csv(file_path)

extract_preds(df, "sarcasm", "irony")
