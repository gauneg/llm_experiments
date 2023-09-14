import os
import argparse
msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--train_key', help='training key to use')
parser.add_argument('--cuda', help='gpu device visible')
args = parser.parse_args()
print(f'Train Dataset: {args.train_key}')
print(f'CUDA DEVICE: {args.cuda}')
# for key in datadict.keys(): 
train_key = args.train_key

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_loader_asp import get_dataset, DataLoaderGen
import numpy as np
import pandas as pd
# import argparse
from prompts import prompt_dict


tokenizer = AutoTokenizer.from_pretrained('/ichec/work/ngeng206c/models/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('/ichec/work/ngeng206c/models/flan-t5-base')

df_pre_lap  =  pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_laptop/final_gen_combine/relabelled.csv')
df_new_lap_2 = pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_188_dep.csv')
df_new_lap_3 = pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_788_dep.csv')

df_pre_res = pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_restaurant/final_gen_combine/relabelled.csv')
df_new_res_2 = pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_restaurant/final_gen_combine/lab_168_dep.csv')
df_new_res_3 = pd.read_csv('/ichec/work/ngeng206c/ds_for_generation/selective_gen_restaurant/final_gen_combine/lab_904_dep.csv')

datadict = {
   'relabelled_lap_188': np.vstack((df_pre_lap.values, df_new_lap_2.values)),
   'relabelled_res_168': np.vstack((df_pre_res.values, df_new_res_2.values)),
   'relabelled_lap_788': np.vstack((df_pre_lap.values, df_new_lap_3.values)),
   'relabelled_res_904': np.vstack((df_pre_res.values, df_new_res_3.values))
}

if __name__ == '__main__':
    print(f'STARTING TRAINING ON:{args.cuda}, USING DS: {train_key}')
    train_dataset = get_dataset(
    DataLoaderGen,
    datadict,
    tokenizer,
    type_path=train_key,
    prompt=prompt_dict['self_for_flan-t5']
    )
                            
    training_args = TrainingArguments(
        output_dir=f"/ichec/work/ngeng206c/models/flan-ate-{train_key}",
        optim='adafactor',
        num_train_epochs=40,
        logging_strategy='steps',
        learning_rate=3e-4,
        logging_steps=100,
        save_strategy='epoch',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        
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
