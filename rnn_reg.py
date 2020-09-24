import torch
from torch import nn
from torch.nn.utils import rnn
from rnn_utils import *


class RNN_reg(nn.Module):
    """
    This model performs vanilla RNN regression between time series. This type of regression is convenient for example
    when dealing with multi-sensor data.
    The input is a multivariate series with dimension input_dim,
    the output is a multivariate series of same length with dimension output_dim.
    This implementation supports batches of padded sequences of variable lengths.
    """
    def __init__(
        self,
        rnn_type,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.0,
        batch_first=True,
    ):
        super(RNN_reg, self).__init__()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(
                self.input_dim,
                self.hidden_dim,
                self.num_layers,
                dropout=dropout,
                batch_first=batch_first,
            )
        else:
            raise ValueError("Invalid RNN type, RNN_reg only supports GRU and LSTM")

        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.hidden = None

    def forward(self, input_, lengths=None):
        """
        Forward pass through RNN layer
        :param input_: multivariate series with dimension input_dim
        :param lengths: list of all the sequences' lengths in the batch, in case input_ is a Tensor containing
        padded sequences of variable length
        :return: multivariate series of same length as input_, with dimension output_dim
        """

        device = input_.get_device()

        batch_size = input_.shape[0]
        h = init_hidden(self.rnn_type, self.num_layers, batch_size, self.hidden_dim, 1, device)

        if lengths is not None:
            packed = rnn.pack_padded_sequence(
                input_, lengths, batch_first=self.batch_first, enforce_sorted=False,
            )
            rnn_out, hidden = self.rnn(packed, h)
            rnn_out, _ = rnn.pad_packed_sequence(
                rnn_out, batch_first=self.batch_first
            )
        else:
            rnn_out, hidden = self.rnn(input_, h)

        # shape of rnn_out: (input_size, batch_size, hidden_dim)
        out = self.linear(rnn_out)
        out = self.sigmoid(out)

        self.hidden = hidden
        # shape of self.hidden: (a, b), a and b shapes: (num_layers, batch_size, hidden_dim).

        return out
