#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--train_key', help='saved_model_name')
parser.add_argument('--cuda', help='gpu device visible')
parser.add_argument('--base_model', help='model to start training from')
args = parser.parse_args()
print(f'Train Dataset: {args.train_key}')
print(f'CUDA DEVICE: {args.cuda}')
import pandas as pd
from asp_pol_dataset import AspectDataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# In[2]:
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
df = pd.read_csv('/home/gauneg/llm_experiments/training_data_original/train_op_polarity.csv')

# In[3]:
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
model_output = os.path.join(f'/home/gauneg/llm_experiments/models/aspect_polarity_prediction/ft_{args.base_model}', args.train_key)
training_args = TrainingArguments(
    output_dir=model_output,
    optim='adafactor',
    num_train_epochs=32,
    logging_strategy='steps',
    learning_rate=3e-4,
    logging_steps=100,
    save_strategy='epoch',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4
)
train_dataset = AspectDataset(df.values, tokenizer)
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