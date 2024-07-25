#!/usr/bin/env python
# coding: utf-8
import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

import json
import timeit
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def convert_type(df):
    for i in range(len(df)):
        df['word'].iloc[i] = eval(df['word'].iloc[i])
        df['labels'].iloc[i] = eval(df['labels'].iloc[i])
    return df

def label_mapping(df):
    dct = {'O':'n','B':'B-T','I':'T'}
    for i in range(len(df)):
        df.labels.iloc[i] = [dct[k] for k in df.labels.iloc[i]]
    df = list(zip(*map(df.get, df)))
    return df

def get_data(trainings_data, val_data, test_data):
    #train
    train_tags=[tup[1] for tup in trainings_data]
    train_texts=[tup[0] for tup in trainings_data]

    #val
    val_tags=[tup[1] for tup in val_data]
    val_texts=[tup[0] for tup in val_data]

    #test
    test_tags=[tup[1] for tup in test_data]
    test_texts=[tup[0] for tup in test_data]
    return train_tags, train_texts, val_tags, val_texts, test_tags, test_texts

# return the extracted terms given the token level prediction and the original texts
def extract_terms(token_predictions, val_texts):
    extracted_terms = set()
    # go over all predictions
    for i in range(len(token_predictions)):
        pred = token_predictions[i]
        txt  = val_texts[i]
        # print(len(pred), len(txt))
        for j in range(len(pred)):
          # if right tag build term and add it to the set otherwise just continue
          # print(pred[j], txt[j])
            if pred[j]=="B-T":
                term=txt[j]
                for k in range(j+1,len(pred)):
                    if pred[k]=="T": term+=" "+txt[k]
                    else: break
                extracted_terms.add(term)
    return extracted_terms


def computeTermEvalMetrics(extracted_terms, gold_df):
    #make lower case cause gold standard is lower case
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set=set(gold_df)
    true_pos=extracted_terms.intersection(gold_set)
    recall=round(len(true_pos)*100/len(gold_set),1)
    precision=round(len(true_pos)*100/len(extracted_terms),1)
    fscore = round(2*(precision*recall)/(precision+recall),1)

    print(str(precision)+' & ' +  str(recall)+' & ' +  str(fscore))
    return len(extracted_terms), len(gold_set), len(true_pos), precision, recall, fscore
