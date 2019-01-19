# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba
from HyperParameter import HyperParameter
import numpy as np
import random
import pickle

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3

zero_pad = np.zeros(768, float)


# print(zero_pad)


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
        self.encoder_bert = []


def load_and_cut_data(filepath):
    """
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    """
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]
            data.append(cutted_line)
    return data


def create_dic_and_map(sources, targets):
    """
    得到输入和输出的字符映射表
    :param sources:
           targets:
    :return: sources_data:
             targets_data:
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    # Load dictionary from file
    hp = HyperParameter()
    with open(hp.dictionary_sources, 'r', encoding='utf-8') as f:
        word_dic_new = f.read().split('\n')

    with open(hp.dictionary_targets, 'r', encoding='utf-8') as f:
        word_dic_new_t = f.read().split('\n')

    # word_dic_new = list(set([character for line in sources for character in line]))

    # word_dic_new_t = list(set([character for line in targets for character in line]))
    # print(len(word_dic_new_t))
    # print(word_dic_new[5])

    # 将字典中的汉字/英文单词映射为数字
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    id_to_word_t = {idx: word for idx, word in enumerate(special_words + word_dic_new_t)}
    word_to_id_t = {word: idx for idx, word in id_to_word_t.items()}
    # print(word_to_id['<GO>'])
    # print(word_to_id_t['.'])

    # print(word_to_id['.'])

    # 将sources和targets中的汉字/英文单词映射为数字
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id_t.get(character, word_to_id_t['<UNK>']) for character in line] for line in targets]
    return sources_data, targets_data, word_to_id, id_to_word, word_to_id_t, id_to_word_t


'''
def create_batch(sources, targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    # len(target) + 1 because of one <EOS>
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)
    init_id = list(range(10))
    random.shuffle(init_id)
    for source in sources:
        # 因为Encoder采用了双向RNN，此处不进行反向，只进行PAD
        # source = list(reversed(source))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(source + pad)
    batch.encoder_inputs=np.array(batch.encoder_inputs)

    #batch.encoder_inputs = batch.encoder_inputs[0]
    batch.encoder_inputs = batch.encoder_inputs[init_id,:].tolist()

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - (len(target) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)
    batch.decoder_targets = np.array(batch.decoder_targets)
    batch.decoder_targets = batch.decoder_targets[init_id, :].tolist()


    return batch
'''


def create_batch(sources, bert_embedding,targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    # len(target) + 1 because of one <EOS>
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:

        pad = [padToken] * (max_source_length - len(source))

        batch.encoder_inputs.append(source + pad)
    #print(type(batch.encoder_inputs))

    for bert in bert_embedding:
        pad = [zero_pad] * (max_source_length - len(bert))
        batch.encoder_bert.append(bert + pad)
        #print(batch.encoder_bert[0])

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - (len(target) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)
    #print(type(batch.decoder_targets))
    return batch

def create_infer_batch(sources, bert_embedding,targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    # len(target) + 1 because of one <EOS>
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = len(bert_embedding[0])
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:

        pad = [padToken] * (max_source_length - len(source))

        batch.encoder_inputs.append(source + pad)
    #print(type(batch.encoder_inputs))

    for bert in bert_embedding:
        pad = [zero_pad] * (max_source_length - len(bert))
        batch.encoder_bert.append(bert + pad)
        #print(batch.encoder_bert[0])

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - (len(target) + 1))
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)
    #print(type(batch.decoder_targets))
    return batch


def get_batches(sources_data, bert_embedding_data, targets_data, batch_size):
    data_len = len(sources_data)

    def gen_next_samples():
        for i in range(0, data_len, batch_size):
            yield sources_data[i:min(i + batch_size, data_len)], bert_embedding_data[
                                                                 i:min(i + batch_size, data_len)], targets_data[
                                                                                                   i:min(i + batch_size,
                                                                                                         data_len)]

    batches = []
    for sources, bert_embedding, targets, in gen_next_samples():
        batch = create_batch(sources, bert_embedding, targets)
        batches.append(batch)

    return batches


def sentence2enco(sentence, word2id):
    """
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，先将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    """
    if sentence == '':
        return None
    # 分词
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    # 将每个单词转化为id
    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    #print(wordIds)
    # 调用createBatch构造batch
    a = pickle.load(open('data/small_bert.txt', 'rb'))
    bbbb =a[9]
    batch = create_infer_batch([wordIds], [bbbb],[[]])
    return batch


if __name__ == '__main__':
    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    keep_rate = 0.6
    batch_size = 32
    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word, word_to_id_t, id_to_word_t = create_dic_and_map(sources,
                                                                                                        targets)
    # create_batch(sources_data, targets_data)
    batches = get_batches(sources_data, bert_embedding_data, targets_data, batch_size)

    # temp = 0
    '''
    for nextBatch in batches:
        if temp == 0:
            print(len(nextBatch.encoder_inputs))
            print(len(nextBatch.encoder_inputs_length))
            print(nextBatch.decoder_targets)
            print(nextBatch.decoder_targets_length)
        temp += 1
    '''
