import pandas as pd 
import numpy as np
import sys


def summarise_liwc(model, train_ds, test_ds, path, to_drop):
    
    path = f'{path}/{model}/{train_ds}/{test_ds}'

    summary = "Analytic	Clout	Authentic	Tone".split()

    


    with open(f'liwc_summary_{model}_{train_ds}_{test_ds}.txt', 'w') as sys.stdout:
        print(path)
        for f in ['fp', 'fn', 'tp', 'tn']:
            data = pd.read_csv(f'{path}/liwc_{f}.csv')
            print(f'for {f} size = {len(data)}')
            print(f'average word count = {np.mean(data["WC"])}')
            print(f'average wps = {np.mean(data["WPS"])}')
            for col in summary:
                print(f'{col} = {np.mean(data[col])} +- {np.std(data[col])}')
            columns = data.columns.drop(to_drop)
            other = data[columns]
            values = []
            for col in other.columns:
                values.append(np.mean(other[col]))
            
            sorted_values = sorted(zip(other.columns, values), key=lambda x: x[1], reverse=True)
            
            print('top 10 features in other columns by average:')
            for i in range(10):
                print(f'{sorted_values[i]} +- {np.std(other[sorted_values[i][0]])}')
            
            print('\n\n')

def summarise_liwc_cart(model, train_ds, test_ds, path):
    to_drop = "label	mean_confidence	variability	mean_correctness	correctnesses".split()

    path = f'{path}/{model}/{train_ds}/{test_ds}'

    summary = "Analytic	Clout	Authentic	Tone".split()

    with open(f'liwc_cart_summary_{model}_{train_ds}.txt', 'w') as sys.stdout:
        print(path)
        for f in ['fp', 'fn', 'tp', 'tn']:
            data = pd.read_csv(f'{path}/liwc_{f}.csv')
            print(f'for {f} size = {len(data)}')
            print(f'average word count = {np.mean(data["WC"])}')
            print(f'average wps = {np.mean(data["WPS"])}')
            for col in summary:
                print(f'{col} = {np.mean(data[col])}')
            columns = data.columns.drop(to_drop)
            other = data[columns]
            values = []
            for col in other.columns:
                values.append(np.mean(other[col]))
            
            sorted_values = sorted(zip(other.columns, values), key=lambda x: x[1], reverse=True)
            
            print('top 10 features in other columns by average:')
            for i in range(10):
                print(sorted_values[i])
            
            print('\n\n')


to_drop = "index	dataset	tweet	label	prediction	probability	Segment	WC	Analytic	Clout	Authentic	Tone	WPS".split()
summarise_liwc('bertweet', 'irony_ds', 'irony_ds', '../preds/crossval4', to_drop)
summarise_liwc('bertweet', 'sarcasm_ds', 'sarcasm_ds', '../preds/crossval4', to_drop)
summarise_liwc('bertweet', 'semeval_mix_ds', 'semeval_mix_ds', '../preds/crossval4', to_drop)
summarise_liwc('roberta', 'sarcasm_ds', 'sarcasm_ds', '../preds/crossval4', to_drop)





# summarise_liwc('roberta', 'semeval_mix', 'irony_ds', 'lowest_correctness/all_predictions', to_drop)

