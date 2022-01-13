import torch
import torch.nn as nn
from stsgcn.torch_utils import zeros, batch_to
from stsgcn.layers import ST_GCNN_layer, CNN_layer


class ZeroVelocity(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = "zero_velocity"
        self.parameter = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)  # dummy parameter

    def forward(self, input):
        """
        Use the last frame of each input sequence as prediction for the test sequence.

        Parameters:
        input (torch.tensor): Input motion sequence with shape (N, T_in, V, C)

        Returns:
        torch.tensor: Output motion sequence with shape (N, T_out, V, C)

        N: number of sequences in the batch
        in_channels: 3D joint coordinates or axis angle representations
        out_channels: 3D joint coordinates or axis angle representations
        T_in: number of frames in the input sequence
        T_out: number of frames in the output sequence
        V: number of joints
        """
        return input[:, input.shape[1]-1:, :, :].repeat(1, self.cfg["output_n"], 1, 1) + self.parameter * 0


class STSGCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = "stsgcn"
        self.joints_to_consider = cfg["joints_to_consider"]
        self.cfg = cfg
        self.input_time_frame = cfg["input_n"]
        self.output_time_frame = cfg["output_n"]
        self.st_gcnns = nn.ModuleList()
        self.input_channels = cfg["input_dim"]
        self.st_gcnn_dropout = cfg["st_gcnn_dropout"]
        self.n_txcnn_layers = cfg["n_tcnn_layers"]
        self.txc_kernel_size = cfg["tcnn_kernel_size"]
        self.txc_dropout = cfg["tcnn_dropout"]
        self.bias = True
        self.txcnns = nn.ModuleList()

        self.st_gcnns.append(
            ST_GCNN_layer(
                self.input_channels, 64, [1, 1], 1, self.input_time_frame, self.joints_to_consider, self.st_gcnn_dropout
            )
        )

        self.st_gcnns.append(
            ST_GCNN_layer(64, 32, [1, 1], 1, self.input_time_frame, self.joints_to_consider, self.st_gcnn_dropout)
        )

        self.st_gcnns.append(
            ST_GCNN_layer(32, 64, [1, 1], 1, self.input_time_frame, self.joints_to_consider, self.st_gcnn_dropout)
        )

        self.st_gcnns.append(
            ST_GCNN_layer(
                64, self.input_channels, [1, 1], 1, self.input_time_frame, self.joints_to_consider, self.st_gcnn_dropout
            )
        )

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(
            CNN_layer(self.input_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
        )  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, self.n_txcnn_layers):
            self.txcnns.append(
                CNN_layer(self.output_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
            )

        self.prelus = nn.ModuleList()
        for j in range(self.n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        """ 
        Parameters:
        input (torch.tensor): Input motion sequence with shape (N, T_in, V, C)
        Returns:
        torch.tensor: Output motion sequence with shape (N, T_out, V, C)
        N: number of sequences in the batch
        in_channels: 3D joint coordinates or axis angle representations
        out_channels: 3D joint coordinates or axis angle representations
        T_in: number of frames in the input sequence
        T_out: number of frames in the output sequence
        V: number of joints
        """

        x = x.permute(0, 3, 1, 2)  # (NTVC->NCTV)

        for gcn in self.st_gcnns:
            x = gcn(x)

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection
        x = x.permute(0, 1, 3, 2)  # (NTCV->NTVC)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', use_bias=True):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh, bias=use_bias))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


class RNN(nn.Module):
    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out


class MotionDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = "stsgcn"
        self.joints_to_consider = cfg["joints_to_consider"]
        self.cfg = cfg
        self.input_time_frame = cfg["input_n"]
        self.output_time_frame = cfg["output_n"]
        self.input_channels = cfg["input_dim"]
        self.n_txcnn_layers = cfg["n_tcnn_layers"]
        self.txc_kernel_size = cfg["tcnn_kernel_size"]
        self.txc_dropout = cfg["tcnn_dropout"]
        self.txcnns = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.joints_to_consider * 3, 1)
        # self.gaussian_noise_std = cfg["gaussian_noise_std"]

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(
            CNN_layer(self.output_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
        )  # with kernel_size[3,3] the dimensions of C,V will be maintained
        self.txcnns.append(
            CNN_layer(self.output_time_frame, 1, self.txc_kernel_size, self.txc_dropout)
        )

    # def add_gaussian_noise(self, y):
    #     return y + self.gaussian_noise_std * torch.randn_like(y)

    def forward(self, y):
        """ 
        Parameters:
        input (torch.tensor): Input motion sequence with shape (N, T_in, V, C)

        Returns:
        torch.tensor: Output motion sequence with shape (N, T_out, V, C)

        N: number of sequences in the batch
        in_channels: 3D joint coordinates or axis angle representations
        out_channels: 3D joint coordinates or axis angle representations
        T_in: number of frames in the input sequence
        T_out: number of frames in the output sequence
        V: number of joints
        """
        # y = self.add_gaussian_noise(y)
        y = y.permute(0, 1, 3, 2)  # (NTVC->NTCV)
        y = self.leaky_relu(self.txcnns[0](y))  # (NTCV)
        y = self.leaky_relu(self.txcnns[1](y))  # (N1CV)
        y = y.view(y.shape[0], -1)  # (N, C*V)
        y = self.sigmoid(self.linear(y))  # (N, 1)
        return y
