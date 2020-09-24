import torch


def init_hidden(rnn_type, num_layers, batch_size, hidden_dim, num_directions, device):
    if rnn_type == "LSTM":
            return (
                torch.zeros(num_directions * num_layers, batch_size, hidden_dim).to(device),
                torch.zeros(num_directions * num_layers, batch_size, hidden_dim).to(device),
            )
    elif rnn_type == "GRU":
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)
    else:
        raise ValueError("Invalid RNN type, only LSTM and GRU are handled")


def repackage_hidden(h, device):
    """Wraps hidden states in new Tensors, to detach them from their history"""
    if isinstance(h, torch.Tensor):
        return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v, device) for v in h)
