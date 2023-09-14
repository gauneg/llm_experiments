from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import ast
from prompts import prompt_dict

class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer):
        self.data = input_data
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.meta_data = []
        self.pol_map = {'pos': 'positive', 
                        'neg': 'negative',
                        'neu': 'neutral',
                        'con': 'conflict',
                        'positive':'positive',
                        'negative':'negative',
                        'neutral':'neutral'}
        
        self._build()

    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids, "target_mask": target_mask, 'meta': self.meta_data[index]}

    def _build(self):
        for idx in range(len(self.data)):
            input_text, aspect_term, aspect_polarity = self.data[idx]            
            input_ = f"{prompt_dict['aspect_sentiment'](aspect_term, input_text)} </s>"
            target = f"{self.pol_map[aspect_polarity]} </s>"
          
            tokenized_inputs = self.tokenizer(
                [input_], padding="max_length", truncation=True, return_tensors="pt"
            )
            tokenized_targets = self.tokenizer(
                [target], padding="max_length", truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
            self.meta_data.append([input_text, aspect_term, self.pol_map[aspect_polarity]])