import torch
from torch.utils.data import DataLoader, Dataset
from dataloader_helper import BIOTagGen, BIOTagGenBPE
import ast
from infer_bert_token_classification import get_labs_from_bio


class AspectBioDataset(Dataset):
    def __init__(self, input_dataset, tokenizer):
        self.data = input_dataset
        self.tokenizer = tokenizer
        
        # for BERT
        # self.labels_to_ids = {
        #     'B': 1,
        #     'I': 2,
        #     'O': 0,
        #     '[PAD]': -100,
        #     '[SEP]': -100,
        #     '[CLS]': -100
        # }
        
        # for  xlm-RoBERTa
        self.labels_to_ids = {
            'B': 1,
            'I': 2,
            'O': 0,
            '<pad>':-100,
            '<s>':-100,
            '</s>':-100
        }
        self.ids_to_labels = {v: k for k, v in self.labels_to_ids.items()}
        self.bio_gen = BIOTagGenBPE(tokenizer)
        self.encoded_data = []
        self._build()

    def encode_text(self, input_text, lab_list):
        lab_parsed = [lab[0] for lab in ast.literal_eval(lab_list)]
        labs, tokenized_inputs, attention_mask = self.bio_gen.prepare_seq_lab_one(input_text, lab_parsed)
        encoded_labs = [self.labels_to_ids[lab] for lab in labs]
        return encoded_labs, tokenized_inputs, attention_mask

    def _build(self):
        for idx in range(len(self.data)):
            input_txt, lab = self.data[idx]
            encoded_labs, tokenized_inputs, attention_mask = self.encode_text(input_txt, lab)
            self.encoded_data.append([encoded_labs, tokenized_inputs, attention_mask])

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        encoded_labs, tokenized_inputs, attention_mask = self.encoded_data[index]
        return {
            "input_ids": torch.tensor(tokenized_inputs),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(encoded_labs)
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import pandas as pd

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    GOLD_LAPTOP = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
    GOLD_RESTAURANT = '/home/gauneg/llm_experiments/csv_test_sets/restaurant_gold_2014.csv'
    TEST_LABS = '/home/gauneg/llm_experiments/csv_test_sets/labelling_test.csv'
    df = pd.read_csv(TEST_LABS)

    asp_bio = AspectBioDataset(df.values, tokenizer)
    with open('/home/gauneg/llm_experiments/model_pred_logs/text_file.txt', 'w') as f:
        for i, data in enumerate(asp_bio):
            inx = data['input_ids']
            labx = data['labels']
            toks = tokenizer.convert_ids_to_tokens(inx)
            f.write(f'input {i}: {inx}')
            f.write(f'labs  {i}: {labx}')
            f.write(f'toks  {i}: {toks}')
            
    # print(asp_bio[0])
