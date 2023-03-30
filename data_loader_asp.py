import pandas as pd
import csv
import os
import torch
FILE_PATH = '/home/gauneg/llm_experiments/ds_for_generation/train_aspect_term.csv'
import ast


dataset =  pd.read_csv(FILE_PATH).sample(frac=1).reset_index(drop=True)
# asp_dset = AspectDataset(dataset,tokenizer)
dataset = dataset.values
val_split = 0.2

INDEX_SPLIT = dataset.shape[0] - int(val_split * dataset.shape[0])
TRAIN_DATASET = dataset[:INDEX_SPLIT, :]
VALID_DATASET = dataset[INDEX_SPLIT:, :]

datadict = {"train": TRAIN_DATASET, "validation": VALID_DATASET}

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer, max_len=512, prompt=""):#, max_len=128) -> None:
        self.data = input_data
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}


    def get_labs(self, lab_tup):
        aterm, _, _, a_entity_category, sentiment = lab_tup
        # print(aterm, sentiment, a_entity_category)
        entity, aspect_category = a_entity_category.split("#")
        return aterm.lower()

    def pre_encode_text(self, input_text, lab_list):
            final_labs = set()
            for lab_tuples in ast.literal_eval(lab_list):
                aterm = self.get_labs(lab_tuples)
                if aterm != 'null':
                    final_labs.add(aterm.lower())
            unencoded_x = self.prompt +' '+input_text.lower()+ ' </s>'
            unencoded_target_str = ', '.join(list(final_labs)) 
            # unencoded_target = 'noaspectterm'+'' if unencoded_target_str.strip()=="" else unencoded_target_str
            return unencoded_x, unencoded_target_str.strip()+ ' </s>'

    def _build(self):
        for idx in range(len(self.data)):
            input_text, lab_list = self.data[idx]
            input_, target = self.pre_encode_text(input_text, lab_list)
            tokenized_inputs = self.tokenizer(
                [input_],  padding="max_length", truncation=True, return_tensors="pt"
            )

            # max_length=self.max_len,
            tokenized_targets = self.tokenizer(
                [target], padding="max_length", truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

    
class MamsAspectDataset(AspectDataset):
    
    def __init__(self, input_data, tokenizer, max_len=512, prompt=''):
        super().__init__(input_data, tokenizer, max_len, prompt=prompt)
    
    def get_labs(self, lab_tuples):
        aterm, _, _, sentiment = lab_tuples
        return aterm


class DataLoaderGen(AspectDataset):

    def __init__(self, input_data, tokenizer, max_len=512, prompt=''):
        super().__init__(input_data, tokenizer, max_len, prompt=prompt)

    def get_labs(self, lab_tuples):
        aterm, _ = lab_tuples
        return aterm

def get_dataset(Dataset, datadict, tokenizer, type_path, prompt=''):
    return Dataset(datadict[type_path], tokenizer, prompt=prompt)
