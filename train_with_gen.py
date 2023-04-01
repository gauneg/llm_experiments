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


df_pre_lab  =  pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/labelled.csv')
df_new_lab_0 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_138_dep.csv')
df_new_lab_1 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_344_dep.csv')
df_new_lab_2 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_592_dep.csv')
df_new_lab_3 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_901_dep.csv')
df_new_lab_4 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_1226_dep.csv')
df_new_lab_5 = pd.read_csv('/home/gauneg/llm_experiments/ds_for_generation/selective_gen_laptop/final_gen_combine/lab_1482_dep.csv')


datadict = {
    'mean_0_sigma':np.vstack((df_pre_lab.values, df_new_lab_0.values)),
    'mean_1_sigma':np.vstack((df_pre_lab.values, df_new_lab_1.values)),
    'mean_2_sigma':np.vstack((df_pre_lab.values, df_new_lab_2.values)),
    'mean_3_sigma':np.vstack((df_pre_lab.values, df_new_lab_3.values)),
    'mean_4_sigma':np.vstack((df_pre_lab.values, df_new_lab_4.values)),
    'mean_5_sigma':np.vstack((df_pre_lab.values, df_new_lab_5.values)),
}


if __name__ == '__main__':
    # for k, v in datadict.items():
    #     print(k, v.shape)


    train_key = 'mean_5_sigma'

    train_dataset = get_dataset(
    DataLoaderGen,
    datadict,
    tokenizer,
    type_path=train_key,
    prompt=prompt_dict['self_for_flan-t5']
    )
                            
    training_args = TrainingArguments(
        output_dir=f"./models/combined_gen_{train_key}_with_prompts",
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
