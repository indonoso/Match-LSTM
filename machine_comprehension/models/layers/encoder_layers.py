__author__ = 'han'

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from ...utils.functions import compute_mask


class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - dataset_h5_path: glove embedding file path
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (seq_len, batch, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding,
                                                  _weight=weights)
        self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        mask = compute_mask(x)

        tmp_emb = self.embedding_layer.forward(x)
        out_emb = tmp_emb.transpose(0, 1)

        return out_emb, mask


class CharEmbedding(torch.nn.Module):
    """
    Char Embedding Layer, random weight
    Args:
        - dataset_h5_path: char embedding file path
    Inputs:
        **input** (batch, seq_len, word_len): word sequence with char index
    Outputs
        **output** (batch, seq_len, word_len, embedding_size): tensor contain char embeddings
        **mask** (batch, seq_len, word_len)
    """

    def __init__(self, dataset_h5_path, embedding_size, trainable=False):
        super(CharEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embedding = self.load_dataset_h5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=embedding_size,
                                                  padding_idx=0)

        # Note that cannot directly assign value. When in predict, it's always False.
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_dataset_h5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            word_dict_size = f.attrs['char_dict_size']

        return int(word_dict_size)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)

        mask = compute_mask(x, 0)  # char-level padding idx is zero
        x_emb = self.embedding_layer.forward(x)
        x_emb = x_emb.view(batch_size, seq_len, word_len, -1)
        mask = mask.view(batch_size, seq_len, word_len)

        return x_emb, mask


class CharEncoder(torch.nn.Module):
    """
    char-level encoder with MyRNNBase used
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(CharEncoder, self).__init__()

        self.encoder = MyStackedRNN(mode=mode,
                                    input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size)
        x = x.transpose(0, 1)
        char_mask = char_mask.view(-1, word_len)

        _, x_encode = self.encoder.forward(x, char_mask)  # (batch*seq_len, hidden_size)
        x_encode = x_encode.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x_encode = x_encode * word_mask.unsqueeze(-1)

        return x_encode.transpose(0, 1)


class CharCNN(torch.nn.Module):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNN, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)

        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)  # (N, 1, word_len, embedding_size)

        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]  # (N, Cout, word_len - fw + 1) * fn
        x = [torch.max(cx, 2)[0] for cx in x]  # (N, Cout) * fn
        x = torch.cat(x, dim=1)  # (N, hidden_size)

        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = x * word_mask.unsqueeze(-1)

        return x.transpose(0, 1)


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size

        self.cnn = CharCNN(emb_size=emb_size,
                           filters_size=filters_size,
                           filters_num=filters_num,
                           dropout_p=dropout_p)

        if enable_highway:
            self.highway = Highway(in_size=hidden_size,
                                   n_layers=2,
                                   dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)

        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)

        return o


class Highway(torch.nn.Module):
    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x
