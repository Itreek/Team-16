import torch
import torch.nn as nn
import numpy as np

# approximation of DHN


class DHN(nn.Module):

    # Initializing all variables
    def __init__(self, element_dim, hidden_dim, target_size, bidirectional, minibatch, is_train=True, sigmoid=True):

        super(DHN, self).__init__()

        # Initializing the parameters
        self.hidden_dim = hidden_dim
        self.bidirect = bidirectional
        self.minibatch = minibatch
        self.sigmoid = sigmoid

        # Initializing the layers
        self.lstm_row = nn.LSTM(
            element_dim, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)
        self.lstm_col = nn.LSTM(
            hidden_dim*2, hidden_dim, bidirectional=self.bidirect, num_layers=2, dropout=0.2)

        if self.bidirect:
            self.hl1 = nn.Linear(hidden_dim * 2, 256)
            self.hl2 = nn.Linear(256, 64)
            self.hl3 = nn.Linear(64, target_size)
        else:
            self.hl1 = nn.Linear(hidden_dim, target_size)

        # Initializing the hidden state
        self.hidden_row = self.init_hidden(1)
        self.hidden_col = self.init_hidden(1)

        # initializing the weights of the layers
        if is_train:
            # initialize the weights in the entire nn.Module recursively for LSTM modules
            for m in self.modules():
                if isinstance(m, nn.LSTM):

                    # The input-to-hidden weight matrix is initialized orthogonally.

                    # 1st Bi-RNN

                    torch.nn.init.orthogonal_(m.w_l1.data)
                    m.bias_ih_l0.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l1_h.data)
                    m.bias_hh_l0.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l1_reverse.data)
                    m.bias_ih_l0_reverse.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l1_h_reverse.data)
                    m.bias_hh_l0_reverse.data[0:self.hidden_dim].fill_(-1)

                    # 2nd Bi-RNN

                    torch.nn.init.orthogonal_(m.w_l2.data)
                    m.bias_ih_l1.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l2_h.data)
                    m.bias_hh_l1.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l2_reverse.data)
                    m.bias_ih_l1_reverse.data[0:self.hidden_dim].fill_(-1)
                    torch.nn.init.orthogonal_(m.w_l2_h_reverse.data)
                    m.bias_hh_l1_reverse.data[0:self.hidden_dim].fill_(-1)

    def init_hidden(self, batch):
        if self.bidirect:
            hidden = np.zeros(2*2, batch, self.hidden_dim)
        else:
            hidden = np.zeros(2, batch, self.hidden_dim)

        return hidden

    def forward(self, X):

        # Initializing hidden row # A tensor of this size is 1-dimensional but has no elements.
        self.hidden_row = self.init_hidden(X.size(0))
        self.hidden_col = self.init_hidden(
            X.size(0))  # Initializing hidden col

        # 1st Bi-RNN

        # flatten the matrix row-wise
        flattened_row = X.view(X.size(0), -1, 1).permute(1, 0, 2).contiguous()
        bi_rnn_1_out, self.hidden_row = self.lstm_row(
            flattened_row, self.hidden_row)

        # reshaping
        bi_rnn_1_out = bi_rnn_1_out.view(-1, bi_rnn_1_out.size(2))
        bi_rnn_1_out = bi_rnn_1_out.view(X.size(1), X.size(2), X.size(0), -1)

        # 2nd Bi-RNN
        flattened_col = bi_rnn_1_out.permute(1, 0, 2, 3).contiguous()
        flattened_col = flattened_col.view(-1, flattened_col.size(2),
                                           flattened_col.size(3)).contiguous()
        bi_rnn_2_out, self.hidden_col = self.lstm_col(
            flattened_col, self.hidden_col)

        # reshaping
        bi_rnn_2_out = bi_rnn_2_out.view(X.size(2), X.size(
            1), X.size(0), -1).permute(1, 0, 2, 3).contiguous()
        bi_rnn_2_out = bi_rnn_2_out.view(-1, bi_rnn_2_out.size(3))

        # 3 fully connected layers
        fc_layer1 = self.hl1(bi_rnn_2_out)
        fc_layer2 = self.hl2(fc_layer1)
        final = self.hl3(fc_layer2).view(-1, X.size(0))

        if self.sigmoid:
            tag_scores = np.sigmoid(final)
        else:
            tag_scores = final

        return tag_scores.view(X.size(1), X.size(2), -1).permute(2, 0, 1).contiguous()
