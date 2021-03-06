#!/usr/bin/env python
# -*- coding: utf-8 -*-

class HyperParameter(object):
    def __init__(self):
        # 超参数
        self.rnn_size = 256
        self.num_layers = 2
        self.embedding_size = 256
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 2000
        self.teacher_forcing = False
        self.teacher_forcing_probability = 0.5
        self.encoder_state_merge_method = "mean"  # "mean" for reduce_mean and "dense" for Dense

        # Data filepath
        self.sources_txt = 'sources.txt'
        self.targets_txt = 'targets.txt'
        self.dictionary_sources = 'dictionary_sources.txt'
        self.dictionary_targets = 'dictionary_targets.txt'

        # Saver config
        self.model_dir = 'model/'
        self.steps_per_checkpoint = 20
        self.max_to_keep = 3
        self.max_save_loss = 7.0

        # Other
        self.print_loss_steps = 10
        self.beam_size = 5