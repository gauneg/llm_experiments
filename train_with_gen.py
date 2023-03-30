import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_loader_asp import get_dataset, DataLoaderGen
import numpy as np
import pandas as pd
from prompts import prompt_dict


tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', max_model_length=256)
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')


train_laptop = '/home/gauneg/llm_experiments/ds_for_generation/laptop_train_twitter_liu_gen.csv'
train_restaurant = '/home/gauneg/llm_experiments/ds_for_generation/restaurant_train_twitter_liu_gen.csv'

df_train_restaurant = pd.read_csv(train_restaurant)
df_train_laptop = pd.read_csv(train_laptop)

# combined_values = np.vstack((df_train_laptop.values, df_train_laptop2.values, mams_train.values))
# np.random.shuffle(combined_values)

datadict = {
   'laptop_train': df_train_laptop.values,

   'restaurant_train': df_train_restaurant.values,

}

# 'combined': np.vstack((df_train.values, df_valid.values))

train_key = 'restaurant_train'

train_dataset = get_dataset(
   DataLoaderGen,
   datadict,
   tokenizer,
   type_path=train_key,
   prompt=prompt_dict['self_for_flan-t5']
)
                         
training_args = TrainingArguments(
    output_dir=f"./models/twitter_liu_{train_key}_with_prompts",
    optim='adafactor',
    num_train_epochs=15,
    logging_strategy='steps',
    learning_rate=5e-5,
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
