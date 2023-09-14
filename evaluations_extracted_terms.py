import argparse
import os
msg = 'argument here'
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--cuda_device', help='gpu device used for training')
parser.add_argument('--model_path', help='path to model to evaluate')

args = parser.parse_args()


cuda_device = args.cuda_device
models_path = args.model_path
mod_name = models_path.split(os.path.sep)[-1]
METRIC_LOG_FILE = f'/ichec/work/ngeng206c/eval_logs/{mod_name}.txt'

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
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



def evaluate_with_dataset(model, tokenizer, dataset, batch_size=32, f_name=""):
    v_dload = DataLoader(dataset, batch_size=batch_size)
    fin_arr = []
    for dpoint in tqdm(v_dload):
        inp_ids = dpoint['input_ids']
        labels = dpoint['labels']
        amask = dpoint['attention_mask']
        y_pred = model.generate(inp_ids.to(model.device))
        
        for x, y, y_true in zip(inp_ids,y_pred, labels):
            pred = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y, skip_special_tokens=True))
            refs = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y_true, skip_special_tokens=True))
            pred_arr = [y.strip().lower() for y in pred.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']
            refs_arr = [y.strip().lower() for y in refs.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']
            x_txt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True))
            fin_arr.append([pred_arr, refs_arr])
    precision, recall, f1 = calc_labs(fin_arr)
    return precision, recall, f1


def infer_with_dataset(model, tokenizer, dataset, batch_size=16, file_out=''):
    out_file_name = f'{file_out}.csv'
    v_dload = DataLoader(dataset, batch_size=batch_size)
    with open(f'/home/gauneg/llm_experiments/model_pred_logs/{out_file_name}', 'w+') as f:
        csv_writer = csv.writer(f, dialect='excel')
        csv_writer.writerow(['text', 'predictions'])
        for dpoint in tqdm(v_dload):
            inp_ids = dpoint['input_ids']
            labels = dpoint['labels']
            amask = dpoint['attention_mask']
            y_pred = model.generate(inp_ids.to(model.device))
            for x, y,y_true in zip(inp_ids,y_pred, labels):
                pred = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y, skip_special_tokens=True))
                refs = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y_true, skip_special_tokens=True))
                pred_arr = [(y.strip().lower(), None) for y in pred.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']
                # refs_arr = [(y.strip().lower(), None) for y in refs.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']
                x_txt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True))
                x_txt = x_txt.replace("Extract aspect terms from the following input. input:","")
                csv_writer.writerow([x_txt.strip(), str(pred_arr)])

def log_res(model_name, ds_name, precision, recall, f1):
    ts = str(datetime.now()).replace(' ', '_')
    with open(METRIC_LOG_FILE, 'a') as f:
        f.write(f'{model_name}/{ds_name} ({ts})  Precision: {precision}, Recall: {recall}, F1: {f1}\n')

if __name__ == '__main__':
    
    LAPTOP_GOLD = '/ichec/work/ngeng206c/ds_for_generation/laptop_train.csv'
    RESTAURANT_GOLD = '/ichec/work/ngeng206c/ds_for_generation/restaurant_train.csv'
    laptop_df = pd.read_csv(LAPTOP_GOLD)
    restaurant_df = pd.read_csv(RESTAURANT_GOLD)

    datadict = {
        'restaurant_gold': restaurant_df.values, 
        'laptop_gold': laptop_df.values
        }

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('/ichec/work/ngeng206c/models/flan-t5-base', max_model_length=256)
    
    for model_dir_name in os.listdir(models_path):
        print(f'Evaluating: {model_dir_name}')
        if model_dir_name != 'runs':
            model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(models_path, model_dir_name))
            model.to(device)
            model.eval()
            for key_id in datadict.keys():
                print(f'Generating metrics for {key_id}')
                validation_dataset = get_dataset(MamsAspectDataset, datadict, tokenizer, type_path=key_id, prompt=prompt_dict['self_for_flan-t5'])
                precise, recall, f1 = evaluate_with_dataset(model, tokenizer, validation_dataset)
                log_res(f'twitter_gen_restaurant_train: {model_dir_name}', key_id+" self_prompt", precise, recall, f1)
        
