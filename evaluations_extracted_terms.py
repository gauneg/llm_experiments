import argparse
import os
from datetime import datetime
msg = 'argument here'
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--cuda_device', help='gpu device used for training')
parser.add_argument('--model_path', help='path to model to evaluate')
# parser.add_argument('--base_model', help='path to the model that was fine-tuned for tokenizer')
args = parser.parse_args()

cuda_device = args.cuda_device
models_path = args.model_path
mod_name = models_path.split(os.path.sep)[-1]
ts = str(datetime.now()).replace(' ', '_')
METRIC_LOG_FILE = f'/home/gauneg/llm_experiments/model_run_logs/last_ate/{mod_name}({ts}).txt'

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
from data_loader_asp import AspectDataset, get_dataset, MamsAspectDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from self_eval import calc_labs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

import csv
import pandas as pd
import numpy as np
from prompts import prompt_dict



def evaluate_with_dataset(model, tokenizer, dataset, batch_size=64, f_name=""):
    v_dload = DataLoader(dataset, batch_size=batch_size)
    fin_arr = []
    file_name = str(datetime.now()).replace(' ', '_')+'__'+f_name
    with open(f'/home/gauneg/llm_experiments/model_pred_logs/{file_name}.txt', 'w+') as f:
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
                f.write(f"{x_txt} -- {pred_arr}, {refs_arr}\n")
    precision, recall, f1 = calc_labs(fin_arr)
    return precision, recall, f1


def infer_with_dataset(model, tokenizer, dataset, batch_size=16, file_out=''):
    out_file_name = f'{file_out}.csv'
    v_dload = DataLoader(dataset, batch_size=batch_size)
    with open(f'/home/gauneg/llm_experiments/ds_for_generation/ft_LLM_gen/{out_file_name}', 'w+') as f:
        csv_writer = csv.writer(f, dialect='excel')
        csv_writer.writerow(['text', 'wlabels'])
        for dpoint in tqdm(v_dload):
            inp_ids = dpoint['input_ids']
            labels = dpoint['labels']
            amask = dpoint['attention_mask']
            y_pred = model.generate(inp_ids.to(model.device))
            for x, y,y_true in zip(inp_ids,y_pred, labels):
                pred = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y, skip_special_tokens=True))
                refs = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y_true, skip_special_tokens=True))
                pred_arr = list(set([(y.strip().lower(), None) for y in pred.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']))
                # refs_arr = [(y.strip().lower(), None) for y in refs.split(',') if y.strip().lower()!='null' and y.strip().lower()!='']
                x_txt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True))
                x_txt = x_txt.replace("Extract aspect terms from the following input. input:","")
                csv_writer.writerow([x_txt.strip(), str(pred_arr)])

def log_res(model_name, ds_name, precision, recall, f1):
    ts = str(datetime.now()).replace(' ', '_')
    with open(METRIC_LOG_FILE, 'a') as f:
        f.write(f'{model_name}/{ds_name} ({ts})  Precision: {precision}, Recall: {recall}, F1: {f1}\n')

if __name__ == '__main__':
    
    LAPTOP_GOLD = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
    RESTAURANT_GOLD = '/home/gauneg/llm_experiments/csv_test_sets/restaurant_gold_2014.csv'
    laptop_df = pd.read_csv(LAPTOP_GOLD)
    restaurant_df = pd.read_csv(RESTAURANT_GOLD)

    datadict = {
        'restaurant_test': restaurant_df.values, 
        'laptop_test': laptop_df.values
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mod = os.listdir(models_path)
    
    sorted_mod_name = sorted([(vx,int(vx.split('-')[-1])) for vx in mod if not vx.startswith('run')], key=lambda x:x[-1])
    mod = [m[0] for m in sorted_mod_name]
    
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    print(f'EVALUATING ')
    
    for model_dir_name in mod:
        print(f'Evaluating: {model_dir_name}')
        if model_dir_name != 'runs':
            model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(models_path, model_dir_name))
            model.to(device)
            model.eval()
            for key_id in datadict.keys():
                print(f'Generating metrics for {key_id}')
                validation_dataset = get_dataset(MamsAspectDataset, datadict, tokenizer, type_path=key_id, prompt=prompt_dict['self_for_flan-t5'])
                precise, recall, f1 = evaluate_with_dataset(model, tokenizer, validation_dataset)
                log_res(f'{model_dir_name}', key_id+" self_prompt", precise, recall, f1)
        
    # model = AutoModelForSeq2SeqLM.from_pretrained('/home/gauneg/llm_experiments/models/base_aspect_term_ft_google/flan-t5-base/checkpoint-1043')
    # model.to(device)
    # model.eval()
    # for key_id in datadict.keys():
    #     print(f'Generating metrics for {key_id}')
    #     train_dataset = get_dataset(MamsAspectDataset, datadict, tokenizer, type_path=key_id, prompt=prompt_dict['self_for_flan-t5'])
    #     infer_with_dataset(model, tokenizer, train_dataset, 64, file_out=key_id)