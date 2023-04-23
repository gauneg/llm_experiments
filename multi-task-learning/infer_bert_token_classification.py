import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForTokenClassification
# from bert_bio_dataloader import AspectBioDataset
import pandas as pd
# from torch.utils.data import Dataset, DataLoader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batches(list_arr, batch_size=8):
    print(list_arr.shape)
    prev_index = 0
    for i in range(batch_size, len(list_arr), batch_size):
        yield list_arr[prev_index: i]
        prev_index = i
    if prev_index < len(list_arr):
        yield list_arr[prev_index:]


def get_predictions(model, tokenizer, text_arr):
    # text array should only be one batch at a time
    batch_inps = get_batches(text_arr)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in batch_inps:
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
            
            print(preds.shape)

            for i in range(len(text_batch)):
                print(i)
            break


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForTokenClassification.from_pretrained('/home/gauneg/llm_experiments/models/bert_bio_aspect/bio-bert-3')
if __name__ == '__main__':
    test_data_path1 = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
    df = pd.read_csv(test_data_path1)
    get_predictions(model, tokenizer, df.values)
