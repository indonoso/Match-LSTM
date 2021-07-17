#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
from ..utils.functions import answer_search
from .layers.answer_point_layers import BoundaryPointer
from .layers.util_layers import MyRNNBase
from .layers.match_layers import MatchRNN


class MatchLSTM(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, hidden_size=150, dropout_p=0.04, emb_dropout_p=0.1, enable_layer_norm=False, hidden_mode='LSTM',
                 embedding_size=300, encoder_bidirection=True, match_lstm_bidirection=True, ptr_bidirection=True):
        super(MatchLSTM, self).__init__()

        encoder_direction_num = 2 if encoder_bidirection else 1

        match_rnn_direction_num = 2 if match_lstm_bidirection else 1

        self.enable_search = True

        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 bidirectional=encoder_bidirection,
                                 dropout_p=emb_dropout_p)
        encode_out_size = hidden_size * encoder_direction_num

        self.match_rnn = MatchRNN(mode=hidden_mode,
                                  hp_input_size=encode_out_size,
                                  hq_input_size=encode_out_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_bidirection,
                                  gated_attention=False,
                                  dropout_p=dropout_p,
                                  enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num

        self.pointer_net = BoundaryPointer(mode=hidden_mode,
                                           input_size=match_rnn_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=ptr_bidirection,
                                           dropout_p=dropout_p,
                                           enable_layer_norm=enable_layer_norm)

    def forward(self, context_vec, context_lengths, question_vec, question_lengths):
        """
        context_char and question_char not used
        """

        # encode: (seq_len, batch, hidden_size)
        context_encode = self.encoder.forward(context_vec, context_lengths)
        question_encode = self.encoder.forward(question_vec, question_lengths)

        # match lstm: (seq_len, batch, hidden_size)
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_lengths,
                                                                               question_encode, question_lengths)
        vis_param = {'match': match_para}

        # pointer net: (answer_len, batch, context_len)
        ans_range_prop = self.pointer_net.forward(qt_aware_ct)
        ans_range_prop = ans_range_prop.transpose(0, 1)

        # answer range
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_lengths)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)

        return ans_range_prop, ans_range, vis_param
