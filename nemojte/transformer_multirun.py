from transformer_main import train_and_evaluate

datasets = ['irony', 'sarcasm', 'mix']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

for i in range(1,2):
    print("======================================= RUN", i, " =======================================")
    for model in models:
        for dataset in datasets:
            train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{i}{params_suffix}", 'irony sarcasm mix', False, False)