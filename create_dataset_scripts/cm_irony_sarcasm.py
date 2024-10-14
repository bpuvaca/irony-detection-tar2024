import numpy as np

def parse_tweet_file(file_path):

    cm_irony = np.zeros((2, 2), dtype=int)
    cm_sarcasm = np.zeros((2, 2), dtype=int)
    cm_both = np.zeros((2,2), dtype=int)

    count_regular_hashS = 0
    count_regular_hashI = 0

    count_neutral = 0
    count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            count += 1
            parts = line.strip().split('\t')
            if len(parts) == 3:
                label = parts[1]
                tweet_text = parts[2]

                #true positive
                if '#irony' in tweet_text.lower() and (label == '1' or label == '2'):
                    cm_irony[0][0] += 1
                #false positive
                if '#irony' not in tweet_text.lower() and (label == '1' or label == '2'):
                    cm_irony[1][0] += 1
                #false negative
                if '#irony' in tweet_text.lower() and (label == '0' or label == '3'):
                    cm_irony[0][1] += 1
                #true negative
                if '#irony' not in tweet_text.lower() and (label == '0' or label == '3'):
                    cm_irony[1][1] += 1

                

                #true positive
                if '#sarcasm' in tweet_text.lower() and label == '3':
                    cm_sarcasm[0][0] += 1
                #false positive
                if '#sarcasm' not in tweet_text.lower() and label == '3':
                    cm_sarcasm[1][0] += 1
                #false negative
                if "#sarcasm" in tweet_text.lower() and label != '3':
                    cm_sarcasm[0][1] += 1
                #true negative
                if "#sarcasm" not in tweet_text.lower() and label != '3':
                    cm_sarcasm[1][1] += 1

                #true irony
                if '#irony' in tweet_text.lower() and (label == '1' or label == '2'):
                    cm_both[0][0] += 1
                #true sarcasm
                if '#sarcasm' in tweet_text.lower() and label == '3':
                    cm_both[1][1] += 1
                #false sarcasm
                if '#irony' in tweet_text.lower() and label == '3':
                    cm_both[0][1] += 1
                #false irony
                if '#sarcasm' in tweet_text.lower() and (label == '1' or label == '2'):
                    cm_both[1][0] += 1

                if '#sarcasm' in tweet_text.lower() and label == '0':
                    count_regular_hashS += 1
                if '#irony' in tweet_text.lower() and label == '0':
                    count_regular_hashI += 1

                if label == '0':
                    count_neutral += 1
                 
    return cm_irony, cm_sarcasm, cm_both, count_regular_hashI, count_regular_hashS, count_neutral, count

file_path = '../datasets/train/SemEval2018-T3-train-taskB_emoji_ironyHashtags.txt'
result = parse_tweet_file(file_path)
print(f"irony: \n{result[0]}")
print(f"sarcasm: \n{result[1]}")

cm_both = result[2]
print(cm_both)
print(f"true irony: {cm_both[0][0]}")
print(f"false irony: {cm_both[1][0]}")
print(f"false sarcasm: {cm_both[0][1]}")
print(f"true sarcasm: {cm_both[1][1]}")


print(f"#irony but neutral: {result[3]}/{result[5]}")
print(f"#sarcasm but neutral: {result[4]}/{result[5]}")
