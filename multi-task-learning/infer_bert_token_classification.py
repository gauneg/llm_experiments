import os
import sys
sys.path.append('/home/gauneg/llm_experiments')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForTokenClassification
# from bert_bio_dataloader import AspectBioDataset
import pandas as pd
# from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import ast
from self_eval import calc_labs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from datetime import datetime


def get_batches(list_arr, batch_size=8):
    prev_index = 0
    for i in range(batch_size, len(list_arr), batch_size):
        yield list_arr[prev_index: i]
        prev_index = i
    if prev_index < len(list_arr):
        yield list_arr[prev_index:]

def get_labs_from_bio(tokens, mask, pred):
    lab_array = []
    has_B = False
    for ix in range(mask.sum() -1):
        lab = pred[ix].item()
        if lab in [1, 2]:
            if lab == 1:
                lab_array.append(tokens[ix])
                has_B = True
            elif lab == 2 and has_B:
                tok = tokens[ix]
                if tok.startswith('##'):
                    lab_array[-1]+=tok.replace('##', '')
                else:
                    lab_array[-1]+=f' {tok}'
    return [item.strip().lower() for item in lab_array]
            


def get_predictions(model, tokenizer, text_arr):
    # text array should only be one batch at a time
    all_preds = []
    batch_inps = get_batches(text_arr)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(batch_inps):
            text_batch = batch[:, 0]
            gold_lab_batch = batch[:, 1]
            tokenized_dataset = tokenizer(text_batch.tolist(), return_tensors='pt', padding='max_length', max_length=512)
            ids = tokenized_dataset['input_ids'].to(device, dtype=torch.long)
            mask = tokenized_dataset['attention_mask'].to(device, dtype=torch.long)
            outputs = model(ids, attention_mask=mask)
            logits = outputs.logits
            active_logits = logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, dim=1)
            preds = torch.argmax(logits, dim=-1)
            
            batch_preds = []
            
            for i in range(len(text_batch)):
                tokens = tokenizer.convert_ids_to_tokens(ids[i])
                pred = preds[i]
                mask_i = mask[i]
                res = get_labs_from_bio(tokens, mask_i, pred)
                batch_preds.append([res, [arr[0].lower() for arr in ast.literal_eval(gold_lab_batch[i])]])
            all_preds += batch_preds
    return all_preds

def write(filepath, res, prefix):
    with open(filepath, 'a') as f:
        f.write(f'{prefix} Precision:{res[0]}, Recall:{res[1]}, F1:{res[2]}\n')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

test_data_path= {
    'restaurant_train_2014' : pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/restaurant_train.csv'),
    'laptop_train_2014' : pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/laptop_train.csv')
}


if __name__ == '__main__':
    checkpoint_path = '/home/gauneg/llm_experiments/models/bio-bert/base_aspect_term'
    outputfile_path = '/home/gauneg/llm_experiments/model_run_logs/bert_like_models'
    file_name = str(datetime.now()).replace(' ', '_')
   
    for ds_name, df  in test_data_path.items():
        result_arr = []

        for i, dir in enumerate(os.listdir(checkpoint_path)):
            model = AutoModelForTokenClassification.from_pretrained(os.path.join(checkpoint_path, dir))
            all_pred_arr = get_predictions(model, tokenizer, df.values)
            precision, recall, f1 = calc_labs(all_pred_arr)
            result_arr.append([dir, precision, recall, f1])
        print(f'DS: {ds_name}, File Name:{file_name}')
        res_df = pd.DataFrame(result_arr, columns=['epochs', 'precision', 'recall', 'f1'])
        res_df.to_csv(os.path.join(outputfile_path, f'{ds_name}({file_name}).csv'), index=False)


