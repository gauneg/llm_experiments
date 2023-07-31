import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class PhraseEncoder:
    def __init__(self,
                 model_path='sentence-transformers/all-MiniLM-L6-v2',
                 tokenizer_path='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.to(self.device)
        self.avg_rep = None
        self.cosine_calc = lambda a,b: np.abs(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_sentences(self, sentence_list):
        encoded_input = self.tokenizer(sentence_list, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device))
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        representation_arr = sentence_embeddings
        # return {sentence_list[i]: representation_arr[i] for i in range(representation_arr.shape[0])}
        return np.array(sentence_list), representation_arr
    
    def get_batch_gen(self, arr, batch_size):
    
        batch_comp = []
        
        for text in arr:
            batch_comp.append(text)

            if len(batch_comp)==batch_size:
                yield batch_comp
                batch_comp = []
    
        if len(batch_comp)>0:
            yield batch_comp
    
    def pre_calc_average(self, arr):
        # text_infer_seq = []
        representaion_seq = []
        lab_enc = self.get_batch_gen(arr, 64)
        for batch in tqdm(lab_enc):
            text, rep = self.encode_sentences(batch)
            # text_infer_seq += text.tolist()
            representaion_seq.append(rep.cpu().detach().numpy())
        all_reps = np.vstack(representaion_seq)
        self.avg_rep = all_reps.mean(axis=0)
    
    def calculate_cosine_sim_avg(self, text):
        sentence_rep = self.encode_sentences([text])[-1].cpu().detach().numpy()
        if self.avg_rep is None:
            raise ValueError('Average not yet calculated, run encode_all')
        return self.cosine_calc(self.avg_rep, sentence_rep[0])
    
    

"""
Current Eval shows:
High precision and low recall, in the dataset which was 
labelled using large language models for out of domain 
dataset.

Hyp:
This is most likely because of the strict criteria that is used in the
selection of the terms in the above model. 
We hypothise that there would be more aspect terms to be found in the 
documents/sentences that are semantically similar to the sentences where
the labels were found.

Idea:
To boost the F1 score, we take the sentences that were unlabelled and rank then
in the order of decreasing semantic similarity with the sentence.
"""