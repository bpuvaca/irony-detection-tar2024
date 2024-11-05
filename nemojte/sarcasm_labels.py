from transformer_main import train_and_evaluate, evaluate_only
import evaluate
datasets = ['isarcasm_sarc']
models = ['roberta', 'bertweet', 'bert']
params_suffix = ''

# for model in models:
#     for dataset in datasets:
#         train_and_evaluate(dataset, model, None, f"{model}_{dataset}_{params_suffix}", None, False, True)

for model in models:
    evaluate_only(model, f"{model}_isarcasm_sarc_", "isarcasm_irony", False, True)