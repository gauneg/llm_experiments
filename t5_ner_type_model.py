#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import csv
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
FILE_PATH = './restaurant_df.csv'
import ast

from transformers import (
    # AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import pytorch_lightning as pl
import torch
import logging
import argparse

dataset =  pd.read_csv(FILE_PATH).sample(frac=1).reset_index(drop=True)
# asp_dset = AspectDataset(dataset,tokenizer)
dataset = dataset.values
val_split = 0.2

INDEX_SPLIT = dataset.shape[0] - int(val_split * dataset.shape[0])
TRAIN_DATASET = dataset[:INDEX_SPLIT, :]
VALID_DATASET = dataset[INDEX_SPLIT:, :]

datadict = {"train": TRAIN_DATASET, "validation": VALID_DATASET}


# In[84]:


class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer):#, max_len=128) -> None:
        self.data = input_data
        self.tokenizer = tokenizer
        # self.max_len = max_len
        # self.tokenizer.max_length = max_len
        # self.tokenizer.model_max_length = max_len
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

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text, lab_list = self.data[idx]
            final_labs = []
            for lab_tuples in ast.literal_eval(lab_list):
                aterm, _, _, a_entity_category, sentiment = lab_tuples
                entity, aspect_category = a_entity_category.split("#")
                # final_labs.append(": ".join([entity.lower(), aspect_category.lower(), aterm.lower()]))
                final_labs.append(aterm.lower())
            input_ = input_text.lower() + ' </s>'
            target = ', '.join(final_labs) + " </s>"
            # print(input_, target)

            # tokenized_inputs = self.tokenizer.batch_encode_plus(
            #     [input_], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            # )
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], padding="max_length", truncation=True, return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], padding="max_length", truncation=True, return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# In[85]:


def get_dataset(tokenizer, type_path, args):
    tokenizer.max_length = args.max_seq_length
    tokenizer.model_max_length = args.max_seq_length
    return AspectDataset(datadict[type_path], tokenizer)


# In[86]:

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparam):
        super(T5FineTuner, self).__init__()
        self.hparam = hparam

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparam.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path
        )
        self.save_hyperparameters()

    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        self.log("train_loss", loss)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                    #    using_native_amp=None,z
                       using_lbfgs=None
                       ):

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam)
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=2)
        t_total = (
            (len(dataloader.dataset) //
             (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
            // self.hparam.gradient_accumulation_steps
            * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)

# In[87]:


logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    
    def on_validation_end(self, trainer, pl_module):
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {} \n".format(key, str(metrics[key])))
    
    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


# In[88]:


# tokenizer = AutoTokenizer.from_pretrained("t5-small",TOKENIZERS_PARALLELISM=True)
tokenizer = AutoTokenizer.from_pretrained("t5-small",TOKENIZERS_PARALLELISM=True)


args_dict = dict(
    # data_dir="wikiann", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=16,
    eval_batch_size=8,
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    # opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args = argparse.Namespace(**args_dict)
model = T5FineTuner(args)


# ### Fine-Tuning T5 for predicting ENTITY: ASPECT: ASPECT TERM

# In[89]:


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename=args.output_dir+"checkpoint.pth", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
     accelerator="gpu", devices=[0],
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    # early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    # amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    callbacks=[checkpoint_callback, LoggingCallback()],
)
trainer = pl.Trainer(**train_params)


# In[90]:

if __name__=='__main__':
    trainer.fit(model)





