import pandas as pd 
import numpy as np
import sys

to_keep = "WC	Analytic	Clout	Authentic	Tone	WPS	BigWords	Dic	Linguistic	function	pronoun	ppron	i	we	you	shehe	they	ipron	det	article	number	prep	auxverb	adverb	conj	negate	verb	adj	quantity	Drives	affiliation	achieve	power	Cognition	allnone	cogproc	insight	cause	discrep	tentat	certitude	differ	memory	Affect	tone_pos	tone_neg	emotion	emo_pos	emo_neg	emo_anx	emo_anger	emo_sad	swear	Social	socbehav	prosocial	polite	conflict	moral	comm	socrefs	family	friend	female	male	Culture	politic	ethnicity	tech	Lifestyle	leisure	home	work	money	relig	Physical	health	illness	wellness	mental	substances	sexual	food	death	need	want	acquire	lack	fulfill	fatigue	reward	risk	curiosity	allure	Perception	attention	motion	space	visual	auditory	feeling	time	focuspast	focuspresent	focusfuture	Conversation	netspeak	assent	nonflu	filler	AllPunc	Period	Comma	QMark	Exclam	Apostro	OtherP	Emoji".split()

def get_features(path, to_keep=to_keep):
    data = pd.read_csv(path)
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == 0]
    data_pos = data_pos[to_keep]
    data_neg = data_neg[to_keep]
    features = {}
    for col in data_pos.columns:
        features[col] = {'pos_mean': np.mean(data_pos[col]), 'neg_mean': np.mean(data_neg[col]), 'pos_std': np.std(data_pos[col]), 'neg_std': np.std(data_neg[col])}
    return features

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
            
def find_useful_features(path, to_keep=to_keep):
    
    features = get_features(path, to_keep)
    useful = [feature for feature in features if abs(features[feature]['pos_mean'] - features[feature]['neg_mean']) > max(features[feature]['pos_std'], features[feature]['neg_std'])]
    
    if len(useful) == 0:
        print(f'No useful features found in {path}')
    else:
        print(f'Useful features found in {path}: {useful}')

            
def get_lowest_std_features(path, to_keep=to_keep, n_best=5):
    features = get_features(path, to_keep)

    sorted_features = sorted(features.keys(), key=lambda x: max(features[x]['pos_std'], features[x]['neg_std']))
    
    print(f'\nTop {n_best} features with lowest std for {path}:')
    for i in range(n_best):
        pos_mean = features[sorted_features[i]]['pos_mean']
        neg_mean = features[sorted_features[i]]['neg_mean']
        pos_std = features[sorted_features[i]]['pos_std']
        neg_std = features[sorted_features[i]]['neg_std']
        print(f'{i + 1}. {sorted_features[i]} ({pos_mean:.2f} +- {pos_std:.2f}) - ({neg_mean:.2f} +- {neg_std:.2f}) = {pos_mean - neg_mean:.2f}')
    
    return sorted_features[:n_best]

            
def get_most_important_features(path, to_keep=to_keep, n_best=5):
    features = get_features(path, to_keep)

    sorted_features = sorted(features.keys(), key=lambda x: features[x]['pos_mean'] - features[x]['neg_mean'], reverse=True)
    
    print(f'\ntop {n_best} features by difference in means for {path}:')
    for i in range(n_best):
        pos_mean = features[sorted_features[i]]['pos_mean']
        neg_mean = features[sorted_features[i]]['neg_mean']
        pos_std = features[sorted_features[i]]['pos_std']
        neg_std = features[sorted_features[i]]['neg_std']
        print(f'{i + 1}. {sorted_features[i]} ({pos_mean:.2f} +- {pos_std:.2f}) - ({neg_mean:.2f} +- {neg_std:.2f}) = {pos_mean - neg_mean:.2f}')
    
    return {key:value for key, value in features.items() if key in sorted_features[:n_best]}


    
    
if __name__ == '__main__':    
    to_drop = "index	dataset	tweet	label	prediction	probability	Segment	WC	Analytic	Clout	Authentic	Tone	WPS".split()
    # summarise_liwc('bertweet', 'irony_ds', 'irony_ds', '../preds/crossval4', to_drop)
    # summarise_liwc('bertweet', 'sarcasm_ds', 'sarcasm_ds', '../preds/crossval4', to_drop)
    # summarise_liwc('bertweet', 'semeval_mix_ds', 'semeval_mix_ds', '../preds/crossval4', to_drop)
    # summarise_liwc('roberta', 'sarcasm_ds', 'sarcasm_ds', '../preds/crossval4', to_drop)

    print("MEANS")
    print(get_most_important_features('../results/LIWC/datasets/liwc_irony_parsed.csv', to_keep))
    get_most_important_features('../results/LIWC/datasets/liwc_sarcasm_parsed.csv', to_keep)
    get_most_important_features('../results/LIWC/datasets/liwc_semeval_mix_parsed.csv', to_keep)
    get_most_important_features('../results/LIWC/datasets/liwc_isarcasm_mix_parsed.csv', to_keep)
    get_most_important_features('../results/LIWC/datasets/liwc_mix_parsed.csv', to_keep)

    print("\nSTD")
    get_lowest_std_features('../results/LIWC/datasets/liwc_irony_parsed.csv', to_keep)
    get_lowest_std_features('../results/LIWC/datasets/liwc_sarcasm_parsed.csv', to_keep)
    get_lowest_std_features('../results/LIWC/datasets/liwc_semeval_mix_parsed.csv', to_keep)
    get_lowest_std_features('../results/LIWC/datasets/liwc_isarcasm_mix_parsed.csv', to_keep)
    get_lowest_std_features('../results/LIWC/datasets/liwc_mix_parsed.csv', to_keep)

    print("\nUSEFUL")
    find_useful_features('../results/LIWC/datasets/liwc_irony_parsed.csv', to_keep)
    find_useful_features('../results/LIWC/datasets/liwc_sarcasm_parsed.csv', to_keep)
    find_useful_features('../results/LIWC/datasets/liwc_semeval_mix_parsed.csv', to_keep)
    find_useful_features('../results/LIWC/datasets/liwc_isarcasm_mix_parsed.csv', to_keep)
    find_useful_features('../results/LIWC/datasets/liwc_mix_parsed.csv', to_keep)


    # summarise_liwc('roberta', 'semeval_mix', 'irony_ds', 'lowest_correctness/all_predictions', to_drop)

