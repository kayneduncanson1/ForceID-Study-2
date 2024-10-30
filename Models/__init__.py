import torch
import torch.nn as nn
import math
from Utils import set_seed

g = set_seed()


class OneF(nn.Module):

    def __init__(self, in_features, out_features):
        super(OneF, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(self.in_features, self.out_features)

    # The cuda param was only used for architectures with an LSTM layer, but is included for all archs so that the
    # function that executed training and validation (train_val func in TrainEval package) could be the same for all
    # architectures.
    def forward(self, inputs, cuda):

        # Reshape input from (N, C, L) to (N, F), where N = number of samples, C = number of channels, L = length, and
        # F = number of features:
        out = inputs.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class TwoF(nn.Module):

    def __init__(self, in_features, fc1_out, out_features):
        super(TwoF, self).__init__()

        self.in_features = in_features
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.fc1 = nn.Sequential(nn.Linear(self.in_features, self.fc1_out), nn.BatchNorm1d(num_features=self.fc1_out),
                                 nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = inputs.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class OneCOneF(nn.Module):

    # nc = number of channels:
    def __init__(self, nc0, nc1, out_features):
        super(OneCOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        # After convolution and average pooling, the sequence length is halved from 200 to 100, so the no. input
        # features for the fully-connected (i.e., linear) layer is nc1 * 100. This is hard-coded below:
        self.fc1 = nn.Linear(self.nc1 * 100, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)
        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class OneCTwoF(nn.Module):

    def __init__(self, nc0, nc1, fc1_out, out_features):
        super(OneCTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.fc1 = nn.Linear(self.nc1 * 100, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out


class ThrCOneF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, out_features):
        super(ThrCOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        # After convolution and average pooling, the sequence length is reduced from 200 to 25, so the no. input
        # features for the fully-connected (i.e., linear) layer is nc1 * 25. This is hard-coded below:
        self.fc1 = nn.Linear(self.nc3 * 25, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class ThrCTwoF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, fc1_out, out_features):
        super(ThrCTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.fc1 = nn.Linear(self.nc3 * 25, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out


class ThrCOneLUOneF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, out_features):
        super(ThrCOneLUOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        # (Params: input_size, hidden_size, num_layers, ...)
        self.rnn = nn.LSTM(self.nc3, self.nc3, 1, bidirectional=False)

        self.fc1 = nn.Linear(self.nc3 * 25, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        # out is (N, C, L) but LSTM input is (L, N, C):
        out = out.view(-1, inputs.size(0), self.nc3)
        h0, c0 = self.init_hidden(inputs, cuda)
        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out

    def init_hidden(self, inputs, cuda):

        h0 = torch.zeros(1, inputs.size(0), self.nc3)
        c0 = torch.zeros(1, inputs.size(0), self.nc3)

        if cuda:

            return [t.cuda() for t in (h0, c0)]

        else:

            return [h0, c0]


class ThrCOneLUTwoF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, fc1_out, out_features):
        super(ThrCOneLUTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.rnn = nn.LSTM(self.nc3, self.nc3, 1, bidirectional=False)

        self.fc1 = nn.Linear(self.nc3 * 25, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        h0, c0 = self.init_hidden(inputs, cuda)
        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out

    def init_hidden(self, inputs, cuda):

        h0 = torch.zeros(1, inputs.size(0), self.nc3)
        c0 = torch.zeros(1, inputs.size(0), self.nc3)

        if cuda:

            return [t.cuda() for t in (h0, c0)]

        else:

            return [h0, c0]


class ThrCOneLBOneF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, out_features):
        super(ThrCOneLBOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        # When using a bi-directional LSTM layer, the no. output features is doubled because of the forward and backward
        # pass:
        self.rnn = nn.LSTM(self.nc3, self.nc3, 1, bidirectional=True)

        # Hence, the no. input features to the fully-connected layer was hard-coded with '* 2':
        self.fc1 = nn.Linear(self.nc3 * 25 * 2, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        h0, c0 = self.init_hidden(inputs, cuda)
        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out

    def init_hidden(self, inputs, cuda):

        h0 = torch.zeros(2, inputs.size(0), self.nc3)
        c0 = torch.zeros(2, inputs.size(0), self.nc3)

        if cuda:

            return [t.cuda() for t in (h0, c0)]

        else:

            return [h0, c0]


class ThrCOneLBTwoF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, fc1_out, out_features):
        super(ThrCOneLBTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.rnn = nn.LSTM(self.nc3, self.nc3, 1, bidirectional=True)

        self.fc1 = nn.Linear(self.nc3 * 25 * 2, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        h0, c0 = self.init_hidden(inputs, cuda)
        out, (hn, cn) = self.rnn(out, (h0, c0))

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out

    def init_hidden(self, inputs, cuda):

        h0 = torch.zeros(2, inputs.size(0), self.nc3)
        c0 = torch.zeros(2, inputs.size(0), self.nc3)

        if cuda:

            return [t.cuda() for t in (h0, c0)]

        else:

            return [h0, c0]


# This generates the positional encoding for inputs to the transformer encoder layer:
class PositionalEncoding(nn.Module):

    # All sequences are fixed length L = 200 (100 frames x 2 stance sides), so max_len param set to 200. This will need
    # to be changed if the pre-processing method is changed and the max sequence length is greater than 200:
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class OneCOneTOneF(nn.Module):

    def __init__(self, nc0, nc1, out_features):
        super(OneCOneTOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.pos_encoder = PositionalEncoding(self.nc1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc1, nhead=4, activation='gelu',
                                                        dim_feedforward=self.nc1 * 100)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(self.nc1 * 100, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)

        # out is (N, C, L) but transformer encoder layer input is (L, N, C):
        out = out.view(-1, inputs.size(0), self.nc1)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class OneCOneTTwoF(nn.Module):

    def __init__(self, nc0, nc1, fc1_out, out_features):
        super(OneCOneTTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.pos_encoder = PositionalEncoding(self.nc1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc1, nhead=4, activation='gelu',
                                                        dim_feedforward=self.nc1 * 100)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(self.nc1 * 100, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)

        out = out.view(-1, inputs.size(0), self.nc1)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out


class ThrCOneTOneF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, out_features):
        super(ThrCOneTOneF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.pos_encoder = PositionalEncoding(self.nc3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc3, nhead=4, activation='gelu',
                                                        dim_feedforward=self.nc3 * 25)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(self.nc3 * 25, self.out_features)

    def forward(self, inputs, cuda):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class ThrCOneTTwoF(nn.Module):

    def __init__(self, nc0, nc1, nc2, nc3, fc1_out, out_features):
        super(ThrCOneTTwoF, self).__init__()

        self.nc0 = nc0
        self.nc1 = nc1
        self.nc2 = nc2
        self.nc3 = nc3
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.conv1 = nn.Sequential(nn.Conv1d(self.nc0, self.nc1, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc1), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv2 = nn.Sequential(nn.Conv1d(self.nc1, self.nc2, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc2), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))
        self.conv3 = nn.Sequential(nn.Conv1d(self.nc2, self.nc3, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(num_features=self.nc3), nn.ELU(),
                                   nn.AvgPool1d(kernel_size=2, stride=2, padding=0))

        self.pos_encoder = PositionalEncoding(self.nc3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.nc3, nhead=4, activation='gelu',
                                                        dim_feedforward=self.nc3 * 25)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.fc1 = nn.Linear(self.nc3 * 25, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):

        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(-1, inputs.size(0), self.nc3)
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out


# Models without convolutions before transformer encoder layer:
class TOneF(nn.Module):

    # num_t_layers = number of transformer encoder layers
    def __init__(self, nc0, num_t_layers, out_features):
        super(TOneF, self).__init__()

        self.nc0 = nc0
        self.num_t_layers = num_t_layers
        self.out_features = out_features

        self.pos_encoder = PositionalEncoding(self.nc0)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.nc0, nhead=3, activation='gelu',
                                                    dim_feedforward=self.nc0 * 200)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_t_layers)

        self.fc1 = nn.Linear(self.nc0 * 200, self.out_features)

    def forward(self, inputs, cuda):

        # reshape from (N, C, L) to (L, N, C):
        out = inputs.reshape(inputs.size(2), inputs.size(0), inputs.size(1))
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)

        return out


class TTwoF(nn.Module):

    def __init__(self, nc0, num_t_layers, fc1_out, out_features):
        super(TTwoF, self).__init__()

        self.nc0 = nc0
        self.num_t_layers = num_t_layers
        self.fc1_out = fc1_out
        self.out_features = out_features

        self.pos_encoder = PositionalEncoding(self.nc0)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.nc0, nhead=3, activation='gelu',
                                                    dim_feedforward=self.nc0 * 200)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_t_layers)

        self.fc1 = nn.Linear(self.nc0 * 200, self.fc1_out)
        self.bn_elu = nn.Sequential(nn.BatchNorm1d(num_features=self.fc1_out), nn.ELU())
        self.fc2 = nn.Linear(self.fc1_out, self.out_features)

    def forward(self, inputs, cuda):
        out = inputs.reshape(inputs.size(2), inputs.size(0), inputs.size(1))
        out = self.pos_encoder(out)
        out = self.transformer_encoder(out)

        out = out.reshape(inputs.size(0), -1)
        out = self.fc1(out)
        out = self.bn_elu(out)
        out = self.fc2(out)

        return out
