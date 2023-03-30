import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from data_loader_asp import AspectDataset, get_dataset, MamsAspectDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from self_eval import calc_labs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from prompts import prompt_dict
from evaluations_extracted_terms import evaluate_with_dataset, log_res


LAPTOP_GOLD_PATH = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
RESTAURANT_GOLD_PATH = '/home/gauneg/llm_experiments/csv_test_sets/restaurant_gold_2014.csv'
df_lap = pd.read_csv(LAPTOP_GOLD_PATH)
df_res = pd.read_csv(RESTAURANT_GOLD_PATH)
datadict = {"laptop_gold": df_lap.values, "restaurant_gold": df_res.values}

dataset_key = 'restaurant_gold'
model_name = 'google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
laptop_test = get_dataset(MamsAspectDataset, datadict, tokenizer, type_path=dataset_key, prompt=prompt_dict['self_for_flan-t5'])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    precision, recall, f1 = evaluate_with_dataset(model, tokenizer, laptop_test)

    log_res(model_name, dataset_key+' self_prompt', precision, recall, f1)