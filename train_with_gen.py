import os
import argparse
msg = "Adding description"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)
parser.add_argument('--train_key', help='training key to use')
parser.add_argument('--cuda', help='gpu device visible')
parser.add_argument('--base_model', help='model to start training from')

args = parser.parse_args()
print(f'Train Dataset: {args.train_key}')
print(f'CUDA DEVICE: {args.cuda}')
# for key in datadict.keys(): 
train_key = args.train_key

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, Adafactor
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from data_loader_asp import get_dataset, DataLoaderGen
import numpy as np
import pandas as pd
# import argparse
from prompts import prompt_dict

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

df_laptop_relabel = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/relabelled.csv')
df_laptop_1 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_13_dep.csv')

df_res_relabel = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_restaurant/final_gen_combine/relabelled.csv')
df_res_1 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_restaurant/final_gen_combine/lab_21_dep.csv')

datadict = {
    # 'base_aspect_term': np.vstack((df_twitter_train.values, df_liu_2009.values, df_liu_2015.values)),
    'laptop_0': np.vstack((df_laptop_relabel.values, df_laptop_1.values)),
    'restaurant_0': np.vstack((df_res_relabel.values, df_res_1.values))
}

if __name__ == '__main__':
    
    print(f'STARTING TRAINING ON:{args.cuda}, USING DS: {train_key}, BASE MODEL:{args.base_model}')
    
    train_dataset = get_dataset(
    DataLoaderGen,
    datadict,
    tokenizer,
    type_path=train_key,
    prompt=prompt_dict['self_for_flan-t5']
    )

    # optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
                            
    training_args = TrainingArguments(
        output_dir=f"./models/select_expriments/{train_key}",
        optim='adafactor',
        num_train_epochs=32,
        logging_strategy='steps',
        learning_rate=3e-4,
        logging_steps=100,
        save_strategy='epoch',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        
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
        train_dataset=train_dataset,
        # optimizers=(optimizer, None)
    )

    trainer.train()