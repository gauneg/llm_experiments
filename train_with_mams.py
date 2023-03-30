import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_loader_asp import get_dataset, MamsAspectDataset
import numpy as np
import pandas as pd
from prompts import prompt_dict


tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', max_model_length=256)
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
# MAMS_TRAIN = '/home/gauneg/llm_experiments/mams_splits/train.csv'
# MAMS_VALID = '/home/gauneg/llm_experiments/mams_splits/valid.csv'
LAPTOP_TRAIN_REAL = '/home/gauneg/llm_experiments/ds_for_generation/laptop_train.csv'
RESTAURANT_TRAIN_REAL = '/home/gauneg/llm_experiments/ds_for_generation/restaurant_train.csv'

df_train_laptop = pd.read_csv(LAPTOP_TRAIN_REAL)
df_train_restaurant = pd.read_csv(RESTAURANT_TRAIN_REAL)

datadict = {'train_laptop': df_train_laptop.values, 'train_restaurant': df_train_restaurant.values}


train_dataset = get_dataset(
   MamsAspectDataset,
   datadict,
   tokenizer,
   type_path='train_restaurant',
   prompt=prompt_dict['self_for_flan-t5']
   )
                         
training_args = TrainingArguments(
    output_dir="./models/supervised_baseline_with_prompt_restaurant2",
    optim='adafactor',
    num_train_epochs=16,
    logging_strategy='steps',
    learning_rate=3e-4,
    logging_steps=100,
    save_strategy='epoch',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
      input_ids = inputs['input_ids']
      labels = inputs['labels'] 
      attention_mask = inputs['attention_mask']
      outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
      return (outputs["loss"], outputs["logits"]) if return_outputs else outputs["loss"]
    

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()