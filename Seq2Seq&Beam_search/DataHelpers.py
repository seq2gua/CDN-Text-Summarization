# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba
from HyperParameter import HyperParameter
import numpy as np
import random
zero_pad = np.zeros(768,float)
padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []

def load_and_cut_data(filepath):


    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]
            data.append(cutted_line)
    return data


def create_dic_and_map(sources, targets):

    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    hp = HyperParameter()
    with open(hp.dictionary_sources, 'r', encoding='utf-8') as f:
        word_dic_new = f.read().split('\n')
        
    with open(hp.dictionary_targets, 'r', encoding='utf-8') as f:
        word_dic_new_t = f.read().split('\n')

    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    id_to_word_t = {idx: word for idx, word in enumerate(special_words + word_dic_new_t)}
    word_to_id_t = {word: idx for idx, word in id_to_word_t.items()}

    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id_t.get(character, word_to_id_t['<UNK>']) for character in line] for line in targets]
    return sources_data, targets_data, word_to_id, id_to_word,word_to_id_t,id_to_word_t


def create_batch(sources, targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(source + pad)

    for target in targets:
        pad = [padToken] * (max_target_length - (len(target) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)

    return batch


def get_batches(sources_data, targets_data, batch_size):
    data_len = len(sources_data)

    def gen_next_samples():
        for i in range(0, data_len, batch_size):
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size,data_len)]

    batches = []
    for sources, targets, in gen_next_samples():
        batch = create_batch(sources, targets)
        batches.append(batch)

    return batches


def sentence2enco(sentence, word2id):
    if sentence == '':
        return None
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    print(wordIds)
    batch = create_batch([wordIds], [[]])
    return batch


if __name__ == '__main__':

    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    keep_rate = 0.6
    batch_size = 128

    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    sources_data, targets_data, word_to_id, id_to_word,word_to_id_t,id_to_word_t = create_dic_and_map(sources, targets)

    batches = get_batches(sources_data, targets_data, batch_size)
