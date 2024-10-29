from transformer_main import train_and_evaluate

datasets = ['iSarcasm_sarc']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

for model in models:
    for dataset in datasets:
        train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{params_suffix}", 'sarcasm from iSarcasm', False, False)