import torch
from torch import nn
from torch.nn.utils import rnn
from rnn_utils import *


class RNNAE(nn.Module):
    """
    From Srivastava et al. 2016 : this RNN autoencoder only uses a fraction of the input time series to reconstruct the
    whole series (therefore, one part of the input is reconstructed, and the other part is predicted). The percentage of
    prediction is a parameter. If pred_pct = 0, RNNAE_pred acts as a vanilla RNN autoencoder.
    Two types of RNN cells are handled : GRU and LSTM
    The decoder splits in two : a reconstructor and a predictor.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="LSTM", dropout=0.0, pred_pct=0.0, ):
        super(RNNAE, self).__init__()
        self.type = rnn_type
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.encode = getattr(nn, rnn_type)(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.reconstruct = getattr(nn, rnn_type + "Cell")(input_dim, hidden_dim, )
        self.predict = getattr(nn, rnn_type + "Cell")(input_dim, hidden_dim, )
        self.linear_rec = nn.Linear(hidden_dim, input_dim)
        self.linear_pred = nn.Linear(hidden_dim, input_dim)
        self.hidden = None
        self.pred_pct = pred_pct

    def decode(self, seq_len, cellstate, decoder, linear):
        """
        Produces either a reconstruction of the input sequence or a prediction of the next seq_len timesteps
        based on the hidden state produced by the encoder.

        Arguments :
        seq_len -- length of the sequence, i.e. number of timesteps to reconstruct
        cellstate -- tuple formed by (hidden state, cell state)
        decoder -- either self.reconstruct or self.predict
        linear -- either self.linear_rec or self.linear_pred
        """

        if self.type == "LSTM":
            hidden, state = cellstate
            hidden, state = hidden[-1], state[-1]
            last_cellstate = (hidden, state)
        elif self.type == "GRU":
            last_cellstate = cellstate[-1]
        else:
            raise ValueError("Invalid RNN type : only LSTM and GRU are handled")
        # we take the hidden representation and cell state from the last layer
        input_ = linear(hidden) if self.type == "LSTM" else linear(last_cellstate)
        out = [input_]
        for i in range(seq_len - 1):
            last_cellstate = decoder(input_, last_cellstate)
            input_ = (
                linear(last_cellstate[0])
                if self.type == "LSTM"
                else linear(last_cellstate)
            )
            out.append(input_)
        return torch.stack(out, dim=1)

    def forward(self, x, lengths=None):
        """
        x has shape (batch_size, seq_len, input_dim)
        lengths is a list of all the sequences' lengths in the batch, in case x is a Tensor containing padded sequences
        of variable length
        """
        seq_len = x.shape[1]
        pred_size = int(self.pred_pct * seq_len)  # nb of timesteps to reconstruct
        if pred_size > 0:
            x = x[:, :-pred_size, :]
        x = x.flip(1)  # the optimization problem is easier to solve if the reconstruction is performed backwards

        batch_size = x.shape[0]
        device = x.get_device()
        h0 = init_hidden(self.type, self.num_layers, batch_size, self.hidden_dim, 1, device)
        # h0 is a tuple containing initial hidden state and/or cell state
        # a hidden/cell state vector has size (num_layers, batch_size, hidden_dim)

        if lengths is not None:
            raise NotImplementedError(
                "This implementation does not take into account yet the different lengths of the sequences in the "
                "batch. To do so we will use a PackedSequence object"
            )
        else:
            _, h = self.encode(x, h0)

        out_rec = self.decode(seq_len - pred_size, h, self.reconstruct, self.linear_rec)
        if pred_size > 0:
            out_pred = self.decode(pred_size, h, self.predict, self.linear_pred)
        else:
            out_pred = None
        # out tensors have shape (batch_size, length, hidden_dim)

        self.hidden = h
        out = (
            torch.cat([out_rec, out_pred], dim=1)
            if not (out_pred is None)
            else out_rec
        )
        return out


class DoubleRNNAE(nn.Module):
    """
    This is a double autoencoder : there are two couples (encoder, decoder). The first
    one is specialized in encoding and decoding the beginnings of the sequences whereas
    the second one is specialized in encoding and decoding the ends of the sequences.
    The input sequence is thus divided in two parts. The first part is reversed before
    being fed to the first encoder. The second part is decoded in reversed order, and
    has to be reversed afterwards. This trick allows a better reconstruction of the first
    points and the last points of the sequence, making the whole reconstruction easier.

    This implementation supports batches of padded sequences of variable lengths. """

    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="LSTM", dropout=0.0):
        super(DoubleRNNAE, self).__init__()
        self.type = rnn_type
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.encode1 = getattr(nn, rnn_type)(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.encode2 = getattr(nn, rnn_type)(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.reconstruct1 = getattr(nn, rnn_type + "Cell")(input_dim, hidden_dim, )
        self.reconstruct2 = getattr(nn, rnn_type + "Cell")(input_dim, hidden_dim, )
        self.linear1 = nn.Linear(hidden_dim, input_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.hidden = None

    def decode(self, seq_len, cellstate, decoder, linear):
        """
        Produces a reconstruction of the 1rst or the 2nd part of the sequence
        based on the hidden state produced by the encoder.

        Arguments :
        seq_len -- length of the sequence, i.e. number of timesteps to reconstruct
        cellstate -- tuple formed by (hidden state, cell state)
        decoder -- either self.reconstruct1 or self.reconstruct2
        linear -- either self.linear1 or self.linear2
        """
        if self.type == "LSTM":
            hidden, state = cellstate
            hidden, state = hidden[-1], state[-1]
            last_cellstate = (hidden, state)
        elif self.type == "GRU":
            last_cellstate = cellstate[-1]
        else:
            raise ValueError("Invalid RNN type : only LSTM and GRU are handled")
        # we take the hidden representation and cell state from the last layer
        input_ = linear(hidden) if self.type == "LSTM" else linear(last_cellstate)
        out = [input_]
        for i in range(seq_len - 1):
            last_cellstate = decoder(input_, last_cellstate)
            input_ = (
                linear(last_cellstate[0])
                if self.type == "LSTM"
                else linear(last_cellstate)
            )
            out.append(input_)
        return torch.stack(out, dim=1)

    def forward(self, x, lengths=None):
        """
        x is divided in 2 parts, the first one is flipped and reconstructed in chronological order,
        while the second one is reconstructed backwards then flipped.
        This allows to achieve good reconstruction at the beginning and at the end of the sequence.
        The 2 reconstructions are eventually concatenated together.

        lengths is a list of all the sequences' lengths in the batch, in case x is a Tensor containing padded sequences
        of variable length
        """
        # input x : (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        device = x.get_device()

        h0 = init_hidden(self.type, self.num_layers, batch_size, self.hidden_dim, 1, device)
        # h0 is a tuple containing initial hidden state and cell state
        # a hidden state vector has size (num_layers, batch_size, hidden_dim)

        seq_len = x.shape[1]
        seq_len1 = seq_len // 2

        if lengths is not None:
            samples1 = []
            lengths1 = [l // 2 for l in lengths]
            samples2 = []
            lengths2 = [l // 2 + l % 2 for l in lengths]
            for b in range(batch_size):
                samples1.append(x[b, :lengths1[b]].flip(1))
                samples2.append(x[b, lengths1[b]:lengths[b]])
            padded1 = rnn.pad_sequence(samples1, batch_first=True)
            padded2 = rnn.pad_sequence(samples2, batch_first=True)
            x1 = rnn.pack_padded_sequence(
                padded1, lengths1, batch_first=True, enforce_sorted=False,
            )
            x2 = rnn.pack_padded_sequence(
                padded2, lengths2, batch_first=True, enforce_sorted=False,
            )
        else:
            x1 = x[:, : seq_len // 2].flip(1)
            x2 = x[:, seq_len // 2:]

        _, h1 = self.encode1(x1, h0)
        _, h2 = self.encode2(x2, h1)

        out1 = self.decode(seq_len1, h1, self.reconstruct1, self.linear1)
        out2 = self.decode(seq_len - seq_len1, h2, self.reconstruct2, self.linear2).flip(1)
        # out tensors have shape (batch_size, length, hidden_dim)

        self.hidden = h2

        if lengths is not None:
            out_samples = []
            for b in range(batch_size):
                o1 = out1[b, :lengths1[b]]
                o2 = out2[b, :lengths2[b]]
                out_samples.append(torch.cat([o1, o2]))
            out = rnn.pad_sequence(out_samples, batch_first=True)

        else:
            out = (
                torch.cat([out1, out2], dim=1)
            )
        return out
