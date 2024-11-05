import pandas as pd

def extract_preds(df):
    
    fp = df[(df['label'] == 0) & (df['prediction'] == 1)]
    fn = df[(df['label'] == 1) & (df['prediction'] == 0)]
    tp = df[(df['label'] == 1) & (df['prediction'] == 1)]
    tn = df[(df['label'] == 0) & (df['prediction'] == 0)]
    print(f'test_examples: {len(df)}\ntp count: {len(tp)}\nfp count: {len(fp)}\nfn count: {len(fn)}\ntn count: {len(tn)}')
    fp.to_csv('train_sarcasm_test_sarcasm_fp.csv', index=False)
    fn.to_csv('train_sarcasm_test_sarcasm_fn.csv', index=False)


# Load the dataframe
# file_path = '../all_preds/bertweet+isarcasm_sarc_test_on_isarcasm_sarc.csv'
# df = pd.read_csv(file_path)

# extract_preds(df)

file_path = '../all_preds/bertweet+isarcasm_sarc_test_on_isarcasm_sarc.csv'
df = pd.read_csv(file_path)

extract_preds(df)
