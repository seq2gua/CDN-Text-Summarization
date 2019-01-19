# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from DataHelpers import *
from model import Seq2SeqModel
from HyperParameter import HyperParameter
import math
import sys
import pickle

if __name__ == '__main__':
    # 超参数
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
    print_loss_steps = hp.print_loss_steps
    beam_size = hp.beam_size
    teacher_forcing = hp.teacher_forcing
    teacher_forcing_probability = hp.teacher_forcing_probability
    max_to_keep = hp.max_to_keep
    max_save_loss = hp.max_save_loss
    encoder_state_merge_method = hp.encoder_state_merge_method

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)
    a = pickle.load(open('data/bert_embedding.txt', 'rb'))


    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word,word_to_id_t,id_to_word_t = create_dic_and_map(sources, targets)
    # Create batch
    bert_embedding_data = a[0:2000]

    batches = get_batches(sources_data, bert_embedding_data,targets_data, batch_size)

    # Train
    with tf.Session() as sess:
        # Build model
        # Note that beam_search should be False while training!!!
        model = Seq2SeqModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            word_to_id=word_to_id,
            word_to_id_t=word_to_id_t,
            mode='train',
            learning_rate=learning_rate,
            use_attention=True,
            beam_search=False,
            beam_size=beam_size,
            encoder_state_merge_method=encoder_state_merge_method,
            teacher_forcing=teacher_forcing,
            teacher_forcing_probability=teacher_forcing_probability,
            max_gradient_norm=5.0,
            max_to_keep=max_to_keep
        )

        # Trying to restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            model.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found, training from scratch...')
            sess.run(tf.global_variables_initializer())

        # Keep track of the minimum loss to save best model
        best_loss = 100000.0
        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            steps = 0
            for nextBatch in batches:
                loss, summary = model.train(nextBatch)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                steps = steps + 1
                if steps % print_loss_steps == 0:
                    print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                # Only save the best model
                if loss < max_save_loss \
                        and loss < best_loss \
                        and steps % steps_per_checkpoint == 0:
                    best_loss = loss
                    model.saver.save(
                        sess, model_dir + 'seq2seq_epoch{}_step{}_loss{:.2f}.ckpt'.format(e, steps, loss)
                    )
            print()
