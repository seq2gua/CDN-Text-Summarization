# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np

from HyperParameter import HyperParameter
from DataHelpers import *
from model import Seq2SeqModel
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_size):

    for single_predict in predict_ids:
        for i in range(beam_size):
            print("Beam search result {}:".format(i + 1))
            predict_list = np.ndarray.tolist(single_predict[:, i])
            predict_seq = [id2word[idx] for idx in predict_list]
            print(" ".join(predict_seq))
            print()


if __name__ == '__main__':

    hp = HyperParameter()
    rnn_size = hp.rnn_size
    num_layers = hp.num_layers
    embedding_size = hp.embedding_size
    batch_size = hp.batch_size
    learning_rate = hp.learning_rate
    epochs = hp.epochs
    steps_per_checkpoint = hp.steps_per_checkpoint
    sources_txt = hp.sources_txt
    targets_txt = hp.targets_txt
    model_dir = hp.model_dir
    beam_size = hp.beam_size
    encoder_state_merge_method = hp.encoder_state_merge_method

    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    sources_data, targets_data, word_to_id, id_to_word,word_to_id_t,id_to_word_t = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        model = Seq2SeqModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            word_to_id=word_to_id,
            word_to_id_t = word_to_id_t,
            mode='predict',
            learning_rate=learning_rate,
            use_attention=True,
            beam_search=True,
            beam_size=beam_size,
            encoder_state_merge_method=encoder_state_merge_method,
            max_gradient_norm=5.0
        )
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(model_dir))

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            batch = sentence2enco(sentence, word_to_id)
            predicted_ids = model.infer(batch)
            predict_ids_to_seq(predicted_ids, id_to_word_t, beam_size)
            sys.stdout.flush()
            sentence = input("> ")
