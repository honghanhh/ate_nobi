import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

from transformers import XLMRobertaTokenizerFast              
from transformers import XLMRobertaForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import logging
from huggingface_hub import HfApi, Repository
import torch
import json
import argparse
import pickle as pkl
from utils import *

logging.set_verbosity_warning()

def convert_type(df):
    for i in range(len(df)):
        df['words'].iloc[i] = eval(df['words'].iloc[i])
        df['labels'].iloc[i] = eval(df['labels'].iloc[i])
    return df

def get_data(trainings_data, val_data, test_data):
    #train
    train_tags = trainings_data['labels'].tolist()
    train_texts= trainings_data['words'].tolist()

    #val
    val_tags = val_data['labels'].tolist()
    val_texts= val_data['words'].tolist()

    #test
    test_tags = test_data['labels'].tolist()
    test_texts= test_data['words'].tolist()
    return train_tags, train_texts, val_tags, val_texts, test_tags, test_texts

def tokenize_and_align_labels(texts, tags):
    # lowercase
    texts = [[x.lower() for x in l] for l in texts]
    tokenized_inputs = tokenizer(
      texts,
      padding=True,
      truncation=True,
      # We use this argument because the texts in our dataset are lists of words (with a label for each word).
      is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs  

# create dataset that can be used for training with the huggingface trainer
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

#compute the metrics TermEval style for Trainer
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    extracted_terms = extract_terms(true_predictions, val) # ??????
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set=gold_validation      # ??????

    # print(extracted_terms)
    true_pos=extracted_terms.intersection(gold_set)
    # print("True pos", true_pos)
    recall=len(true_pos)/len(gold_set)
    precision=len(true_pos)/len(extracted_terms)
    f1 = 2*(precision*recall)/(precision+recall) if precision + recall != 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def extract_terms(token_predictions, val_texts):
    extracted_terms = set()
    # go over all predictions
    for i in range(len(token_predictions)):
        pred = token_predictions[i]
        txt  = val_texts[i]
        # print(len(pred), len(txt))
        for j in range(len(pred)):
            if pred[j]=="BN-Term" or pred[j]=="IN-Term":
                extracted_terms.add(txt[j])
          # if right tag build term and add it to the set otherwise just continue
            if pred[j]=="B-Term" or pred[j]=="BN-Term":
                # print(pred[j], txt[j])
                term=txt[j]
                for k in range(j+1,len(pred)):
                    if pred[k]=="I-Term" or pred[k]=="IN-Term":
                        term+=" "+txt[k]
                        # print(pred[k], txt[k], term)
                    else: break
                extracted_terms.add(term)
    return extracted_terms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train1", default="train1", type=str, required=False)
    parser.add_argument("--train2", default="train2", type=str, required=False)
    parser.add_argument("--val", default="val", type=str, required=False)
    parser.add_argument("--test", default="test", type=str, required=False)
    parser.add_argument("--gold_val", default="gold_val", type=str, required=False)
    parser.add_argument("--gold_test", default="gold_test", type=str, required=False)
    parser.add_argument("--output_dir", default="output_dir", type=str, required=False)
    parser.add_argument("--log_dir", default="log", type=str, required=False)
    parser.add_argument("--metric_path", default="metric", type=str, required=False)
    parser.add_argument("--prediction_path", default="prediction", type=str, required=False)
    parser.add_argument("--hf_repo_id", type=str, required=True, help="The ID of the Hugging Face Hub repository to push to.")
    
    args = parser.parse_args()
        
    train1 = pd.read_csv(args.train1)
    train2 = pd.read_csv(args.train2)
    trainings_data = convert_type(pd.concat([train1, train2]))
    val_data = convert_type(pd.read_csv(args.val))
    test_data = convert_type(pd.read_csv(args.test))

    gold_set_for_validation= pd.read_csv(args.gold_val, header=None, delimiter='\t')[0].tolist()
    gold_set_for_test   = pd.read_csv(args.gold_test, header=None, delimiter='\t')[0].tolist()

    train_tags, train_texts, val_tags, val_texts, test_tags, test_texts = get_data(trainings_data, val_data, test_data)


    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    #align labels with tokenization from XLM-R

    label_list=['O', 'B-Term', 'BN-Term', 'IN-Term', 'I-Term']
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels=len(label_list)

    train_input_and_labels = tokenize_and_align_labels(train_texts, train_tags)
    val_input_and_labels = tokenize_and_align_labels(val_texts, val_tags)
    test_input_and_labels = tokenize_and_align_labels(test_texts, test_tags)

    train_dataset = OurDataset(train_input_and_labels, train_input_and_labels["labels"])
    val_dataset = OurDataset(val_input_and_labels, val_input_and_labels["labels"])
    test_dataset = OurDataset(test_input_and_labels, test_input_and_labels["labels"])

    start = timeit.default_timer()

    training_args = TrainingArguments(
        output_dir= args.output_dir,          # output directory
        num_train_epochs=20,              # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        learning_rate=2e-5,
        logging_dir= args.log_dir,            # directory for storing logs
        evaluation_strategy= 'steps', # or use epoch here
        eval_steps = 500,
        load_best_model_at_end=True,   #loads the model with the best evaluation score
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=True,              # enable pushing to the hub
        hub_model_id=args.hf_repo_id,  # specify the Hugging Face repo ID
        hub_strategy="end",            # push at the end of training
    )

    # initialize model
    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)

    val = val_texts
    gold_validation =  gold_set_for_validation

    # initialize huggingface trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

    # train
    trainer.train()

    # Save model to Hugging Face Hub
    trainer.push_to_hub()

    stop = timeit.default_timer()
    print('Time: ', stop - start) 

    val = test_texts
    gold_validation =  gold_set_for_test

    #test
    test_predictions, test_labels, test_metrics = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_predictions, axis=2)
    # Remove ignored index (special tokens)
    true_test_predictions = [
        [label_list[p] for (p, l) in zip(test_prediction, test_label) if l != -100]
        for test_prediction, test_label in zip(test_predictions, test_labels)
    ]

    test_extracted_terms = extract_terms(true_test_predictions, test_texts)
    extracted, gold, true_pos, precision, recall, fscore = computeTermEvalMetrics(test_extracted_terms, set(gold_set_for_test))
    with open(args.metric_path, 'w') as f:
        f. write(json.dumps([[extracted, gold, true_pos, precision, recall, fscore]]))

    with open(args.prediction_path, 'w') as f:
        f. write(json.dumps(list(test_extracted_terms)))
