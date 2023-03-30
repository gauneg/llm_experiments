import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from data_loader_asp import AspectDataset, get_dataset, MamsAspectDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from self_eval import calc_labs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from datetime import datetime
import csv
import pandas as pd
import numpy as np
from prompts import prompt_dict
from evaluations_extracted_terms import infer_with_dataset

if __name__ == '__main__':
    LAPTOP_TRAIN = '/home/gauneg/llm_experiments/ds_for_generation/laptop_train.csv'
    RESTAURANT_TRAIN = '/home/gauneg/llm_experiments/ds_for_generation/restaurant_train.csv'
    laptop_df = pd.read_csv(LAPTOP_TRAIN)
    restaurant_df = pd.read_csv(RESTAURANT_TRAIN)

    # datadict = {'train': df_train.values, 'valid': df_valid.values, 'combined': np.vstack((df_train.values, df_valid.values))}
    datadict = {'laptop_train': laptop_df.values, 'restaurant_train':restaurant_df.values}
    model = AutoModelForSeq2SeqLM.from_pretrained('/home/gauneg/llm_experiments/models/all_twitter_liu_with_prompts/checkpoint-10430')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', max_model_length=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.eval()
    
    for dkey in datadict.keys():
        laptop_train_dataset = get_dataset(MamsAspectDataset, datadict, tokenizer, type_path=dkey, prompt=prompt_dict['self_for_flan-t5'])
        infer_with_dataset(model, tokenizer, laptop_train_dataset, batch_size=16, file_out=dkey+"_twitter_liu_gen")