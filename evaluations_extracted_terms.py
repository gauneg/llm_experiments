import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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

METRIC_LOG_FILE = './models/experiment_run_history.txt'

def evaluate_with_dataset(model, tokenizer, dataset, batch_size=32, f_name=""):
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
    
    LAPTOP_GOLD = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
    RESTAURANT_GOLD = '/home/gauneg/llm_experiments/csv_test_sets/restaurant_gold_2014.csv'
    laptop_df = pd.read_csv(LAPTOP_GOLD)
    restaurant_df = pd.read_csv(RESTAURANT_GOLD)

    datadict = {
        'restaurant_gold': restaurant_df.values, 
        'laptop_gold': laptop_df.values
        }

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_path = '/home/gauneg/llm_experiments/models/twitter_liu_restaurant_train_with_prompts'
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', max_model_length=256)
    
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
                log_res(f'twitter_gen_restaurant: {model_dir_name}', key_id+" self_prompt", precise, recall, f1)
        