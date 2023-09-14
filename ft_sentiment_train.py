#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--cuda', help='gpu device visible')
parser.add_argument('--base_model', help='model to start training from')
parser.add_argument('--dataset_key', help='model to start training from')

args = parser.parse_args()
print(f'CUDA DEVICE: {args.cuda}')
import pandas as pd
from asp_pol_dataset import AspectDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
dataset_key = args.dataset_key

# In[2]:
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
# default training
# df = pd.read_csv('/ichec/work/ngeng206c/llm_experiments/training_data_original/train_op_polarity.csv')


tokenizer = AutoTokenizer.from_pretrained(args.base_model)

laptop_df = pd.read_csv('/ichec/home/users/gauneg/llm_experiments/ds_for_generation/sentiment_weak_lab_gen/flan-t5-senti-best/laptop_sent_gen.csv')
restaurant_df = pd.read_csv('/ichec/home/users/gauneg/llm_experiments/ds_for_generation/sentiment_weak_lab_gen/flan-t5-senti-best/restaurant_sent_gen.csv')

laptop_df = laptop_df[['text', 'aspect_term', 'polarity']]
restaurant_df = restaurant_df[['text', 'aspect_term', 'polarity']]

dataset_dict_gen = {
   'laptop_gen_train': AspectDataset(laptop_df.values, tokenizer),
   'restaurant_gen_train': AspectDataset(restaurant_df.values, tokenizer)
}


model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
model_output = f'/ichec/work/ngeng206c/models/asp_sentiment_pol/flan-t5-base-asp-sentiment_{dataset_key}'
training_args = TrainingArguments(
    output_dir=model_output,
    optim='adafactor',
    num_train_epochs=32,
    logging_strategy='steps',
    learning_rate=3e-4,
    logging_steps=100,
    save_strategy='epoch',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8
)

train_dataset = dataset_dict_gen[dataset_key]

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