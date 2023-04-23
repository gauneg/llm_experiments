import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModelForTokenClassification
from bert_bio_dataloader import AspectBioDataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch import cuda
from sklearn.metrics import accuracy_score

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_data_path = '/home/gauneg/llm_experiments/ds_for_generation/train_aspect_term_edit.csv'
test_data_path1 = '/home/gauneg/llm_experiments/csv_test_sets/laptop_gold_2014.csv'
test_data_path2 = '/home/gauneg/llm_experiments/csv_test_sets/restaurant_gold_2014.csv'

train_data = pd.read_csv(train_data_path)
test_data_lap = pd.read_csv(test_data_path1)
test_data_res = pd.read_csv(test_data_path2)

training_set = AspectBioDataset(train_data.values, tokenizer)
test_set_lap = AspectBioDataset(test_data_lap.values, tokenizer)
test_set_rest = AspectBioDataset(test_data_res.values, tokenizer)

TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
MAX_GRAD_NORM = 10
LEARNING_RATE = 1e-05
EPOCHS = 10

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_params = {
    'batch_size': TEST_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

train_dataloader = DataLoader(training_set, **train_params)
testing_dataloader_lap = DataLoader(test_set_lap, **test_params)
testing_dataloader_rest = DataLoader(test_set_rest, **test_params)
n_labs = len(training_set.labels_to_ids)
model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=n_labs)
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    for idx, batch in enumerate(train_dataloader):

        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        # print(ids.shape, mask.shape, labels.shape)
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        tr_logits = outputs.logits
        # print(loss, tr_logits)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, dim=1)  # shape (batch_size * seq_len,)

        active_accuracy = labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        train(epoch)
        if epoch>0 and epoch%3==0:
            model.save_pretrained(f'/home/gauneg/llm_experiments/models/bio-bert-{epoch}')