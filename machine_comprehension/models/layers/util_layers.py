__author__ = 'han'

import torch
import torch.nn.functional as F
from ...utils.functions import masked_softmax


class MyStackedRNN(torch.nn.Module):
    """
    RNN with packed sequence and dropout, multi-layers used
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: number of rnn layers
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyStackedRNN, self).__init__()
        self.num_layers = num_layers
        self.rnn_list = torch.nn.ModuleList([MyRNNBase(mode, input_size, hidden_size, bidirectional, dropout_p,
                                                       enable_layer_norm) for _ in range(num_layers)])

    def forward(self, v, mask):
        v_last = None
        for i in range(self.num_layers):
            v, v_last = self.rnn_list[i].forward(v, mask)

        return v, v_last


class AttentionPooling(torch.nn.Module):
    """
    Attention-Pooling for pointer net init hidden state generate.
    Equal to Self-Attention + MLP
    Modified from r-net.
    Args:
        input_size: The number of expected features in the input uq
        output_size: The number of expected features in the output rq_o

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (batch, output_size): tensor containing the output features
    """

    def __init__(self, input_size, output_size):
        super(AttentionPooling, self).__init__()

        self.linear_u = torch.nn.Linear(input_size, output_size)
        self.linear_t = torch.nn.Linear(output_size, 1)
        self.linear_o = torch.nn.Linear(input_size, output_size)

    def forward(self, uq, mask):
        q_tanh = F.tanh(self.linear_u(uq))
        q_s = self.linear_t(q_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        alpha = masked_softmax(q_s, mask, dim=1)  # (batch, seq_len)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)) \
            .squeeze(1)  # (batch, input_size)

        rq_o = F.tanh(self.linear_o(rq))  # (batch, output_size)
        return rq_o


class SelfAttentionGated(torch.nn.Module):
    """
    Self-Attention Gated layer, it`s not weighted sum in the last, but just weighted
    math: \softmax(W*\tanh(W*x)) * x

    Args:
        input_size: The number of expected features in the input x

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output
        - **output** (seq_len, batch, input_size): gated output tensor
    """

    def __init__(self, input_size):
        super(SelfAttentionGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        g_tanh = F.tanh(self.linear_g(x))
        gt = self.linear_t.forward(g_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        gt_prop = masked_softmax(gt, x_mask, dim=1)
        gt_prop = gt_prop.transpose(0, 1).unsqueeze(2)  # (seq_len, batch, 1)
        x_gt = x * gt_prop

        return x_gt


class SelfGated(torch.nn.Module):
    """
    Self-Gated layer. math: \sigmoid(W*x) * x
    """

    def __init__(self, input_size):
        super(SelfGated, self).__init__()

        self.linear_g = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        x_l = self.linear_g(x)  # (seq_len, batch, input_size)
        x_gt = F.sigmoid(x_l)

        x = x * x_gt

        return x


class SeqToSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h1: (seq1_len, batch, hidden_size)
        - h1_mask: (batch, seq1_len)
        - h2: (seq2_len, batch, hidden_size)
        - h2_mask: (batch, seq2_len)
    Outputs:
        - output: (seq1_len, batch, hidden_size)
        - alpha: (batch, seq1_len, seq2_len)
    """

    def __init__(self):
        super(SeqToSeqAtten, self).__init__()

    def forward(self, h1, h2, h2_mask):
        h1 = h1.transpose(0, 1)
        h2 = h2.transpose(0, 1)

        alpha = h1.bmm(h2.transpose(1, 2))  # (batch, seq1_len, seq2_len)
        alpha = masked_softmax(alpha, h2_mask.unsqueeze(1), dim=2)  # (batch, seq1_len, seq2_len)

        alpha_seq2 = alpha.bmm(h2)  # (batch, seq1_len, hidden_size)
        alpha_seq2 = alpha_seq2.transpose(0, 1)

        return alpha_seq2, alpha


class SelfSeqAtten(torch.nn.Module):
    """
    Args:
        -
    Inputs:
        - h: (seq_len, batch, hidden_size)
        - h_mask: (batch, seq_len)
    Outputs:
        - output: (seq_len, batch, hidden_size)
        - alpha: (batch, seq_len, seq_len)
    """

    def __init__(self):
        super(SelfSeqAtten, self).__init__()

    def forward(self, h, h_mask):
        h = h.transpose(0, 1)
        batch, seq_len, _ = h.shape

        alpha = h.bmm(h.transpose(1, 2))  # (batch, seq_len, seq_len)

        # make element i==j to zero
        mask = torch.eye(seq_len, dtype=torch.uint8, device=h.device)
        mask = mask.unsqueeze(0)
        alpha.masked_fill_(mask, 0.)

        alpha = masked_softmax(alpha, h_mask.unsqueeze(1), dim=2)
        alpha_seq = alpha.bmm(h)

        alpha_seq = alpha_seq.transpose(0, 1)
        return alpha_seq, alpha


class SFU(torch.nn.Module):
    """
    only two input, one input vector and one fusion vector

    Args:
        - input_size:
        - fusions_size:
    Inputs:
        - input: (seq_len, batch, input_size)
        - fusions: (seq_len, batch, fusions_size)
    Outputs:
        - output: (seq_len, batch, input_size)
    """

    def __init__(self, input_size, fusions_size):
        super(SFU, self).__init__()

        self.linear_r = torch.nn.Linear(input_size + fusions_size, input_size)
        self.linear_g = torch.nn.Linear(input_size + fusions_size, input_size)

    def forward(self, input, fusions):
        m = torch.cat((input, fusions), dim=-1)

        r = F.tanh(self.linear_r(m))  # (seq_len, batch, input_size)
        g = F.sigmoid(self.linear_g(m))  # (seq_len, batch, input_size)
        o = g * r + (1 - g) * input

        return o


class MemPtrNet(torch.nn.Module):
    """
    memory pointer net
    Args:
        - input_size: zs and hc size
        - hidden_size:
        - dropout_p:
    Inputs:
        - zs: (batch, input_size)
        - hc: (seq_len, batch, input_size)
        - hc_mask: (batch, seq_len)
    Outputs:
        - ans_out: (ans_len, batch, seq_len)
        - zs_new: (batch, input_size)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(MemPtrNet, self).__init__()

        self.start_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.start_sfu = SFU(input_size, input_size)
        self.end_net = ForwardNet(input_size=input_size * 3, hidden_size=hidden_size, dropout_p=dropout_p)
        self.end_sfu = SFU(input_size, input_size)

        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, hc, hc_mask, zs):
        hc = self.dropout(hc)

        # start position
        zs_ep = zs.unsqueeze(0).expand(hc.size())  # (seq_len, batch, input_size)
        x = torch.cat((hc, zs_ep, hc * zs_ep), dim=-1)  # (seq_len, batch, input_size*3)
        start_p = self.start_net(x, hc_mask)  # (batch, seq_len)

        us = start_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)  # (batch, input_size)
        ze = self.start_sfu(zs, us)  # (batch, input_size)

        # end position
        ze_ep = ze.unsqueeze(0).expand(hc.size())
        x = torch.cat((hc, ze_ep, hc * ze_ep), dim=-1)
        end_p = self.end_net(x, hc_mask)

        ue = end_p.unsqueeze(1).bmm(hc.transpose(0, 1)).squeeze(1)
        zs_new = self.end_sfu(ze, ue)

        ans_out = torch.stack([start_p, end_p], dim=0)  # (ans_len, batch, seq_len)

        # make sure not nan loss
        new_mask = 1 - hc_mask.unsqueeze(0).type(torch.uint8)
        ans_out.masked_fill_(new_mask, 1e-6)

        return ans_out, zs_new


class ForwardNet(torch.nn.Module):
    """
    one hidden layer and one softmax layer.
    Args:
        - input_size:
        - hidden_size:
        - output_size:
        - dropout_p:
    Inputs:
        - x: (seq_len, batch, input_size)
        - x_mask: (batch, seq_len)
    Outputs:
        - beta: (batch, seq_len)
    """

    def __init__(self, input_size, hidden_size, dropout_p):
        super(ForwardNet, self).__init__()

        self.linear_h = torch.nn.Linear(input_size, hidden_size)
        self.linear_o = torch.nn.Linear(hidden_size, 1)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x, x_mask):
        h = F.relu(self.linear_h(x))
        h = self.dropout(h)
        o = self.linear_o(h)
        o = o.squeeze(2).transpose(0, 1)  # (batch, seq_len)

        beta = masked_softmax(o, x_mask, dim=1)
        return beta


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, lengths):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        v_sort = v.index_select(1, idx_sort)
        lengths_sort.cpu()
        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len

        return o_unsort
