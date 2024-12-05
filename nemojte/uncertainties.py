import numpy as np
import pandas as pd

def aleatoric_uncertainty(probs):
    means = np.mean(probs, axis=0)
    means0 = np.mean(1-probs, axis=0)
    return -means*np.log(means+1e-9) - means0*np.log(means0+1e-9)
    
def epistemic_uncertainty(probs):
    var1 = np.var(probs, axis=0)
    var0 = np.var(1-probs, axis=0)
    return 1/2*(var1 + var0)

def add_uncertainties(df, column_names, new_column_name):
    probs = []
    for column in column_names:
        probs.append(df[column].values)
    probs = np.array(probs)
    aleatoric = aleatoric_uncertainty(probs)
    epistemic = epistemic_uncertainty(probs)
    df[new_column_name + "_alea"] = aleatoric
    df[new_column_name + "_epi"] = epistemic
    df[new_column_name + "_total"] = aleatoric + epistemic

df = pd.read_csv('ALL_PROBABILITIES_4.csv')
bert_columns = [col for col in df.columns if col.startswith('bert_')]
roberta_columns = [col for col in df.columns if col.startswith('roberta_')]
bertweet_columns = [col for col in df.columns if col.startswith('bertweet_')]

add_uncertainties(df, bert_columns, 'bert')
add_uncertainties(df, roberta_columns, 'roberta')
add_uncertainties(df, bertweet_columns, 'bertweet')

df.to_csv('ALL_PROBABILITIES_AND_UNCERTAINTIES_4.csv', index=False)

