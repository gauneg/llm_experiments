import os
import argparse
msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)

parser.add_argument('--cuda', help='gpu device visible')
parser.add_argument('--model_size', help='model to start training from')

args = parser.parse_args()

mod_size = args.model_size

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

from asp_pol_dataset import AspectDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from self_eval import calc_labs
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from prompts import prompt_dict
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
# from evaluations_extracted_terms import evaluate_with_dataset, log_res
from datetime import datetime

def convert_ids_text(id_arr, tokenizer):
    # print(id_arr)
    toks = tokenizer.convert_ids_to_tokens(id_arr, skip_special_tokens=True)
    return tokenizer.convert_tokens_to_string(toks).strip()

LAPTOP_GOLD_PATH = '/home/gauneg/hester_letter/all_combined_ds/aspect_sentiment_polarity/laptop_2014/test.csv'
RESTAURANT_GOLD_PATH = '/home/gauneg/hester_letter/all_combined_ds/aspect_sentiment_polarity/restaurant_2014/test.csv'
BATCH_SIZE = 16
df_lap = pd.read_csv(LAPTOP_GOLD_PATH)
df_res = pd.read_csv(RESTAURANT_GOLD_PATH)


tokenizer = AutoTokenizer.from_pretrained(f'google/flan-t5-{mod_size}')
datadict = {
    "laptop_gold": DataLoader(AspectDataset(df_lap.values, tokenizer), batch_size=BATCH_SIZE), 
    "restaurant_gold":  DataLoader(AspectDataset(df_res.values, tokenizer), batch_size=BATCH_SIZE)
    }


def infer_res(dataloader, model, tokenizer):
    fin_lab_pred = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        gold_labels = batch['labels']
        pred_labels = model.generate(input_ids.to(device))
        for i in range(input_ids.shape[0]):
            input_txt = convert_ids_text(input_ids[i], tokenizer)
            pred_txt = convert_ids_text(pred_labels[i], tokenizer)
            gold_txt = convert_ids_text(gold_labels[i], tokenizer)
            fin_lab_pred.append([input_txt, pred_txt, gold_txt])    
    return fin_lab_pred 



if __name__ == '__main__':

    # CODE FOR ZERO SHOT
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # OUTPUT_PATH = '/home/gauneg/llm_experiments/model_run_logs'

    # for k, dloader in datadict.items():
    #     res_arr = infer_res(dloader, model, tokenizer)
    #     df_res = pd.DataFrame(res_arr, columns=['text', 'pred_y', 'true_y'])
    #     ts = str(datetime.now()).replace(' ', '_')
        
    #     file_name = f'zero_shot_{k}_pol_large({ts}).txt'
    #     with open(os.path.join(OUTPUT_PATH, file_name), 'w+') as f:
    #         f.write(str(classification_report(df_res['true_y'], df_res['pred_y'], zero_division=0)))
    #         f.write('\n')
    

    model_dir = f'/home/gauneg/llm_experiments/models/aspect_polarity_prediction/ft_google/flan-t5-{mod_size}/twitter_gadget_{mod_size}'
    
    for k, dloader in datadict.items():
        metric_arr = []
        for mod in os.listdir(model_dir):
            if mod != 'runs':
                model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_dir, mod))
                res_arr = infer_res(dloader, model, tokenizer)
                df_res = pd.DataFrame(res_arr, columns=['text', 'pred_y', 'true_y'])
                metric_arr.append([mod, 
                                precision_score(df_res['true_y'], df_res['pred_y'], zero_division=0, average='weighted'), 
                                recall_score(df_res['true_y'], df_res['pred_y'], zero_division=0, average='weighted'), 
                                f1_score(df_res['true_y'], df_res['pred_y'], zero_division=0, average='weighted')
                                ])
        df_scores = pd.DataFrame(metric_arr, columns=['checkpoint', 'precision', 'recall', 'f1'])
        df_scores.to_csv(f'/home/gauneg/llm_experiments/model_run_logs/polarity_only_res/{mod_size}_{k}.csv', index=False)

                


