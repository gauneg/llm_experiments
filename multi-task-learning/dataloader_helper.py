import ast

import torch
from transformers import AutoTokenizer


class BIOTagGen:
    def __init__(self, pre_trained_tokenizer):
        self.tokenizer = pre_trained_tokenizer

    def tokenizer_with_torch(self, phrase):
        tk = self.tokenizer(phrase, max_length=512, padding='max_length')

        toks = self.tokenizer.convert_ids_to_tokens(tk['input_ids'])
        conv_toks = [self.tokenizer.convert_tokens_to_string([tok]).strip().lower() for tok in toks]
        # print(phrase, conv_toks)
        return conv_toks, tk['input_ids'], tk['attention_mask']

    @staticmethod
    def filter_match(arr):
        return all([elem > -1 for elem in arr])

    def sequence_matcher_pos_phrase(self, text_tokens, aspect_phrase):
        phrase_word_seq, tk_input_ids, tk_input_attention_mask = self.tokenizer_with_torch(aspect_phrase)
        # print(phrase_word_seq, tk_input_ids, tk_input_attention_mask)
        last_match = len(tk_input_attention_mask) -1 - tk_input_attention_mask[::-1].index(1)
        phrase_word_seq = phrase_word_seq[1: last_match]
        tagged_sent = text_tokens
        # print(aspect_phrase, phrase_word_seq)
        matched_indices = [-1] * len(phrase_word_seq)
        for i_sent, word in enumerate(tagged_sent):
            if -1 not in matched_indices:
                break
            for i_phrase, phrase_word in enumerate(phrase_word_seq):
                if word == phrase_word:
                    # print(word, phrase_word, i_phrase, i_sent, len(matched_indices))
                    if i_phrase == 0:
                        matched_indices = [-1] * len(phrase_word_seq)
                        matched_indices[i_phrase] = i_sent
                    else:
                        if matched_indices[i_phrase - 1] > -1 and matched_indices[i_phrase - 1] + 1 == i_sent:
                            matched_indices[i_phrase] = i_sent
                        # else:
                        #     matched_indices = [-1] * len(phrase_word_seq)
        return matched_indices

    def prepare_seq_lab_one(self, text_inp: str, lab_phrases: list, labelling_scheme='bio'):
        """
        Labels
        """
        input_tokens_trunc,  tk_input_ids, tk_input_attention_mask = self.tokenizer_with_torch(text_inp)
        unique_labs = set(lab_phrases)
        lab_list = [self.sequence_matcher_pos_phrase(input_tokens_trunc, lab) for lab in list(unique_labs)]
        # print(unique_labs, lab_list)
        lab_list = [lab_seq for lab_seq in lab_list if self.filter_match(lab_seq)]

        bert_special = ['[PAD]', '[SEP]', '[CLS]']
        
        tags_sentence = [(word, 'O') if word not in bert_special else (word, word) for word in input_tokens_trunc]
        if labelling_scheme == 'bio':
            for lab_seq in lab_list:
                for i, lab_index in enumerate(lab_seq):
                    # print(lab_index)
                    if i == 0:
                        tags_sentence[lab_index] = (tags_sentence[lab_index][0], "B")
                    else:
                        tags_sentence[lab_index] = (tags_sentence[lab_index][0], "I")
        return [tag for word, tag in tags_sentence], tk_input_ids, tk_input_attention_mask


class BIOTagGenBPE(BIOTagGen):
    def __init__(self, pre_trained_tokenizer):
        super(BIOTagGenBPE, self).__init__(pre_trained_tokenizer)

    def prepare_seq_lab_one(self, text_inp: str, lab_phrases: list, labelling_scheme='bio'):
        """
        Labels
        """
        input_tokens_trunc,  tk_input_ids, tk_input_attention_mask = self.tokenizer_with_torch(text_inp)
        unique_labs = set(lab_phrases)
        lab_list = [self.sequence_matcher_pos_phrase(input_tokens_trunc, f' {lab}') for lab in list(unique_labs)]
        print('lab_list', len(lab_list), lab_list)
        lab_list = [lab_seq for lab_seq in lab_list if self.filter_match(lab_seq)]
        
        bpe_special = ['<pad>', '<s>', '</s>']
        tags_sentence = [(word, 'O') if word not in bpe_special else (word, word) for word in input_tokens_trunc]

        if labelling_scheme == 'bio':
            for lab_seq in lab_list:
                for i, lab_index in enumerate(lab_seq):
                    # print(lab_index)
                    if i == 0:
                        tags_sentence[lab_index] = (tags_sentence[lab_index][0], "B")
                    else:
                        tags_sentence[lab_index] = (tags_sentence[lab_index][0], "I")
        return [tag for word, tag in tags_sentence], tk_input_ids, tk_input_attention_mask

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    GOLD_LAPTOP = '/Users/gauneg/datasets/aspect_mining/aspect_mine_review/csv_test_sets/laptop_gold_2014.csv'
    GOLD_RESTAURANT = '/Users/gauneg/datasets/aspect_mining/aspect_mine_review/csv_test_sets/restaurant_gold_2014.csv'

    df = pd.read_csv(GOLD_LAPTOP)

    random_index = np.random.randint(0, df.shape[0], 50)
    """
    We have a two fold requirement here for the task of multi-task learning:
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bio_gen = BIOTagGen(tokenizer)

    text = "I re-seated the ""WLAN"" card inside and re-installed the LAN device drivers."
    lab = "[('""WLAN"" card', '27', '16', 'neutral'), ('LAN device drivers', '74', '56', 'neutral')]"
    par_lab = [lab_i[0] for lab_i in ast.literal_eval(lab)]
    print(par_lab)
    print(bio_gen.prepare_seq_lab_one(text, par_lab))

    # for text, lab in df.values:
    #     par_lab = [lab_i[0] for lab_i in ast.literal_eval(lab)]
    #     print(par_lab)
    #     print(bio_gen.prepare_seq_lab_one(text, par_lab))
