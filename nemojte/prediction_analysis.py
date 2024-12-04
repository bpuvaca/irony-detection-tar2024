import pandas as pd
import numpy as np

df = pd.read_csv('ALL_PREDICTIONS_5.csv')

def false_positives(df, threshold_fp):
    column_names = df.columns[3:]
    labeled_0 = df[df['label'] == 0]
    false_positives = labeled_0[labeled_0[column_names].sum(axis=1) > threshold_fp]
    print(f"False positive count (>{threshold_fp} wrong): {len(false_positives)}")
    print("False positives:")
    for fp in false_positives.itertuples():
        print(f"Dataset: {fp.dataset}, Tweet: {fp.tweet}")
    print()

def false_negatives(df, threshold_fn):
    column_names = df.columns[3:]
    model_count = len(column_names)
    labeled_1 = df[df['label'] == 1]
    false_negatives = labeled_1[labeled_1[column_names].sum(axis=1) < (model_count - threshold_fn)]
    print(f"False negative count (>{threshold_fn} wrong): {len(false_negatives)}")
    print("False negatives:")
    for fn in false_negatives.itertuples():
        print(f"Dataset: {fn.dataset}, Tweet: {fn.tweet}")
    print()

def true_positives(df, threshold_tp):
    column_names = df.columns[3:]
    labeled_1 = df[df['label'] == 1]
    true_positives = labeled_1[labeled_1[column_names].sum(axis=1) > threshold_tp]
    print(f"True positive count (>{threshold_tp} correct): {len(true_positives)}")
    print("True positives:")
    for tp in true_positives.itertuples():
        print(f"Dataset: {tp.dataset}, Tweet: {tp.tweet}")
    print()

def true_negatives(df, threshold_tn):
    column_names = df.columns[3:]
    model_count = len(column_names)
    labeled_0 = df[df['label'] == 0]
    true_negatives = labeled_0[labeled_0[column_names].sum(axis=1) < (model_count - threshold_tn)]
    print(f"True negative count (>{threshold_tn} correct): {len(true_negatives)}")
    print("True negatives:")
    for tn in true_negatives.itertuples():
        print(f"Dataset: {tn.dataset}, Tweet: {tn.tweet}")
    print()

#false_positives(df, 11)
false_negatives(df, 11)
#true_positives(df, 11)
#true_negatives(df, 11)