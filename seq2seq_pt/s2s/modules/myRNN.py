import torch
import torch.nn as nn
from torch.autograd import Variable
import s2s.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

try:
    import ipdb
except ImportError:
    pass


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.linear_hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden, mask=None):
        x_W = self.linear_input(input)
        h_U = self.linear_hidden(hidden)
        x_Ws = x_W.split(self.hidden_size, 1)
        h_Us = h_U.split(self.hidden_size, 1)
        r = self.sigmoid(x_Ws[0] + h_Us[0])
        z = self.sigmoid(x_Ws[1] + h_Us[1])
        h1 = self.tanh(x_Ws[2] + r * h_Us[2])
        h = (h1 - hidden) * z + hidden
        if mask:
            h = (h - hidden) * mask.unsqueeze(1).expand_as(hidden) + hidden
        return h

    def __repr__(self):
        return self.__class__.__name__ + '({0}, {1})'.format(self.input_size, self.hidden_size)
