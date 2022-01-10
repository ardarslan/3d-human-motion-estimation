import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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


class STSGCN_Attention(nn.Module):
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
        # self.n_txcnn_layers = cfg["n_tcnn_layers"]
        # self.txc_kernel_size = cfg["tcnn_kernel_size"]
        # self.txc_dropout = cfg["tcnn_dropout"]
        self.bias = True
        # self.txcnns = nn.ModuleList()
        self.scheduled_sampling_target_number = 1.0

        self.h = cfg["n_attention_heads"]
        self.d = self.joints_to_consider * self.input_channels
        self.Wq = nn.Linear(self.d, self.d * self.h, bias=False)
        self.Wk = nn.Linear(self.d, self.d * self.h, bias=False)
        self.Wv = nn.Linear(self.d, self.d * self.h, bias=False)
        self.gru = torch.nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d * self.h, self.d)

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
        # self.txcnns.append(
        #     CNN_layer(self.input_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
        # )  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        # for i in range(1, self.n_txcnn_layers):
        #     self.txcnns.append(
        #         CNN_layer(self.output_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
        #     )
        # self.prelus = nn.ModuleList()
        # for j in range(self.n_txcnn_layers):
        #     self.prelus.append(nn.PReLU())

    def forward(self, x, y=None):
        """ 
        Parameters:
        x (torch.tensor): Input motion sequence with shape (N, T_in, V, C)
        y (torch.tensor): Label motion sequence with shape (N, T_out, V, C)  or  None

        Returns:
        torch.tensor: Output motion sequence with shape (N, T_out, V, C)

        N: number of sequences in the batch
        in_channels: 3D joint coordinates or axis angle representations
        out_channels: 3D joint coordinates or axis angle representations
        T_in: number of frames in the input sequence
        T_out: number of frames in the output sequence
        V: number of joints
        """
        N, T_in, V, C = x.shape
        T_out = self.output_time_frame

        x = x.permute(0, 3, 1, 2)  # (NTVC->NCTV)

        for gcn in self.st_gcnns:
            x = gcn(x)  # NCTV

        # x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        # x = self.prelus[0](self.txcnns[0](x))

        # for i in range(1, self.n_txcnn_layers):
        #     x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection
        # x = x.permute(0, 1, 3, 2)  # (NTCV->NTVC)
        if y is not None:
            y = y.view(N, T_out, V*C)

        x = x.permute(0, 2, 3, 1)  # (NCTV->NTVC)
        x = x.contiguous().view(N, T_in, V*C)  # ( NTVC->NT(V*C) )

        h = self.h
        d = V*C
        queries = self.Wq(x).view(N, T_in, h, V*C).transpose(1, 2).contiguous().view(N*h, T_in, d)  # (N*h, T_in, V*C)
        keys = self.Wk(x).view(N, T_in, h, V*C).transpose(1, 2).contiguous().view(N*h, T_in, d)  # (N*h, T_in, V*C)
        values = self.Wv(x).view(N, T_in, h, V*C).transpose(1, 2).contiguous().view(N*h, T_in, d)  # (N*h, T_in, V*C)
        current_weights = F.softmax(torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d), dim=-1)  # (N*h, T_in, T_in)
        current_weighted_values = torch.bmm(current_weights, values).view(N*h, T_in, V*C)  # (N*h, T_in, V*C)

        yhats = []
        _, current_hidden_state = self.gru(current_weighted_values)  # (2, N*h, V*C)
        current_hidden_state_mean = current_hidden_state.mean(dim=0)  # (N*h, V*C)
        current_yhat = self.linear(current_hidden_state_mean.view(N, h*V*C)) + x[:, -1, :]  # (N, V*C), residual connection
        yhats.append(current_yhat)

        for i in range(T_out-1):
            random_number = np.random.rand()
            if y is None or random_number > self.scheduled_sampling_target_number:  # feed predictions as input
                current_input = current_yhat.unsqueeze(1).detach()  # (N, 1, V*C)
            else:
                current_input = y[:, i, :].unsqueeze(1)  # (N, 1, V*C)
            current_query = self.Wq(current_input).view(N, 1, h, V*C).transpose(1, 2).contiguous().view(N*h, 1, d)  # (N*h, 1, V*C)
            current_key = self.Wk(current_input).view(N, 1, h, V*C).transpose(1, 2).contiguous().view(N*h, 1, d)  # (N*h, 1, V*C)
            current_value = self.Wv(current_input).view(N, 1, h, V*C).transpose(1, 2).contiguous().view(N*h, 1, d)  # (N*h, 1, V*C)
            current_weights = F.softmax(torch.bmm(current_query, keys.transpose(1, 2)) / np.sqrt(d), dim=-1)  # (N*h, 1, T)
            current_weighted_value = torch.bmm(current_weights, values).view(N*h, 1, V*C)  # (N*h, 1, V*C)
            _, current_hidden_state = self.gru(current_weighted_value, current_hidden_state)  # (2, N*h, V*C)
            current_hidden_state_mean = current_hidden_state.mean(dim=0)  # (N*h, V*C)
            current_yhat = self.linear(current_hidden_state_mean.view(N, h*V*C)) + current_yhat  # (N, V*C), residual connection
            yhats.append(current_yhat)
            keys = torch.cat([keys, current_key], dim=1)
            values = torch.cat([values, current_value], dim=1)

        yhats = torch.stack(yhats, dim=1)  # (N, T_out, (V*C))
        yhats = yhats.view(N, T_out, V, C)  # (N, T_out, V, C)
        return yhats


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
        self.gaussian_noise_std = cfg["gaussian_noise_std"]

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(
            CNN_layer(self.output_time_frame, self.output_time_frame, self.txc_kernel_size, self.txc_dropout)
        )  # with kernel_size[3,3] the dimensions of C,V will be maintained
        self.txcnns.append(
            CNN_layer(self.output_time_frame, 1, self.txc_kernel_size, self.txc_dropout)
        )
    
    def add_gaussian_noise(self, y):
        return y + self.gaussian_noise_std * torch.randn_like(y)

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
        y = self.add_gaussian_noise(y)
        y = y.permute(0, 1, 3, 2)  # (NTVC->NTCV)
        y = self.leaky_relu(self.txcnns[0](y))  # (NTCV)
        y = self.leaky_relu(self.txcnns[1](y))  # (N1CV)
        y = y.view(y.shape[0], -1)  # (N, C*V)
        y = self.sigmoid(self.linear(y))  # (N, 1)
        return y


"""
class MotionDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nx = nx = cfg["nx"]
        self.ny = ny = cfg["ny"]
        self.nz = cfg["nz"]
        self.nh_rnn = nh_rnn = cfg.get('nh_rnn', 256)
        self.nh_mlp = nh_mlp = cfg.get('nh_mlp', [300, 150, 1])
        self.horizon = horizon = cfg["output_n"]
        self.rnn_type = rnn_type = cfg.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = cfg.get('x_birnn', True)
        self.e_birnn = e_birnn = cfg.get('e_birnn', True)
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp, use_bias=True)
        self.linear = nn.Linear(cfg["output_n"], 1)

        self.freqbasis = cfg.get('freqbasis', 'dct')
        if self.freqbasis == 'dct':
            dct_mat = torch.FloatTensor(sio.loadmat('../dct_matrices/dct_{:d}.mat'.format(horizon))['D'])
            self.register_buffer('dct_mat', dct_mat)
        elif self.freqbasis == 'dct_adaptive':
            dct_mat = nn.Parameter(torch.tensor(sio.loadmat('../dct_matrices/dct_50.mat')['D']))
            self.register_parameter('dct_mat', dct_mat)

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)
        return h_y

    def forward(self, x, y):
        x_reshaped = x.permute(1, 0, 2, 3)  # (T, N, V, C)
        x_reshaped = x_reshaped.view(x_reshaped.shape[0], x_reshaped.shape[1], x_reshaped.shape[2]*x_reshaped.shape[3])  # (T, N, V*C)

        y_reshaped = y.permute(1, 0, 2, 3)  # (T, N, V, C)
        y_reshaped = y_reshaped.view(y_reshaped.shape[0], y_reshaped.shape[1], y_reshaped.shape[2]*y_reshaped.shape[3])  # (T, N, V*C)

        h_x = self.encode_x(x_reshaped)
        h_y = self.encode_y(y_reshaped)
        h = torch.cat((h_x.repeat(h_y.shape[0], 1, 1), h_y), dim=-1)  # [t,b,d]
        if 'dct' in self.freqbasis:
            h = torch.einsum('wt,tbd->wbd', self.dct_mat, h)  # [w, b, d]
        elif 'dft' in self.freqbasis:
            h = torch.rfft(h.permute(1, 2, 0), 1, onesided=False).permute(2, 3, 0, 1)

        result = self.e_mlp(h)
        result = result.permute(1, 0, 2)[:, :, 0]
        result = self.linear(result)
        return result
"""

"""
class VAEDCT(nn.Module):
    def __init__(self, cfg):
        super(VAEDCT, self).__init__()
        self.nx = nx = cfg["nx"]
        self.ny = ny = cfg["ny"]
        self.nz = nz = cfg["nz"]
        self.horizon = horizon = cfg["output_n"]
        self.rnn_type = rnn_type = cfg.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = cfg.get('x_birnn', True)
        self.e_birnn = e_birnn = cfg.get('e_birnn', True)
        self.use_drnn_mlp = cfg.get('use_drnn_mlp', False)
        self.residual = cfg.get('residual', False)
        self.nh_rnn = nh_rnn = cfg.get('nh_rnn', 256)
        self.nh_mlp = nh_mlp = cfg.get('nh_mlp', [300, 200])
        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.q_bias = cfg.get('posteriorbias', True)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp, use_bias=self.q_bias)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz, bias=self.q_bias)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz, bias=self.q_bias)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')

        # load freq matrix
        self.freqbasis = cfg.get('freqbasis', 'dct')
        if self.freqbasis == 'dct':
            dct_mat = torch.FloatTensor(sio.loadmat('../dct_matrices/dct_{:d}.mat'.format(horizon))['D'])
            self.register_buffer('dct_mat', dct_mat)
        elif self.freqbasis == 'dct_adaptive':
            dct_mat = nn.Parameter(torch.tensor(sio.loadmat('../dct_matrices/dct_50.mat')['D']))
            self.register_parameter('dct_mat', dct_mat)
        elif self.freqbasis == 'dft':
            self.d_rnn = RNN(ny + 2*nz + nh_rnn, nh_rnn, cell_type=rnn_type)
            self.d_rnn.set_mode('step')

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)
        return h_y

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x.repeat(h_y.shape[0], 1, 1), h_y), dim=-1)  # [t,b,d]
        if 'dct' in self.freqbasis:
            h = torch.einsum('wt,tbd->wbd', self.dct_mat, h)  # [w, b, d]
        elif 'dft' in self.freqbasis:
            h = torch.rfft(h.permute(1, 2, 0), 1, onesided=False).permute(2, 3, 0, 1)

        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if 'dct' in self.freqbasis:
            z = torch.einsum('tw,wbd->tbd', self.dct_mat.T, z)
        elif 'dft' in self.freqbasis:
            z = torch.ifft(z.permute(2, 3, 0, 1), 1).permute(2, 0, 1, 3)  # [t,b,d,2]
            z = z.reshape(z.shape[0], z.shape[1], -1)  # [t,b,2d]
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[1], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[1])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z[i], y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            if self.residual:
                y_i += y_p
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y):
        x_reshaped = x.permute(1, 0, 2, 3)  # (T, N, V, C)
        x_reshaped = x_reshaped.view(x_reshaped.shape[0], x_reshaped.shape[1], x_reshaped.shape[2]*x_reshaped.shape[3])  # (T, N, V*C)

        y_reshaped = y.permute(1, 0, 2, 3)  # (T, N, V, C)
        y_reshaped = y_reshaped.view(y_reshaped.shape[0], y_reshaped.shape[1], y_reshaped.shape[2]*y_reshaped.shape[3])  # (T, N, V*C)

        mu, logvar = self.encode(x_reshaped, y_reshaped)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(x_reshaped, z)
        return decoded, mu, logvar

    def sample_prior(self, x, mode='iid'):
        x_reshaped = x.permute(1, 0, 2, 3)  # (T, N, V, C)
        x_reshaped = x_reshaped.view(x_reshaped.shape[0], x_reshaped.shape[1], x_reshaped.shape[2]*x_reshaped.shape[3])  # (T, N, V*C)

        if mode == 'iid':
            if 'dft' in self.freqbasis:
                z = np.random.randn(self.horizon, 2, x_reshaped.shape[1], self.nz)
            else:
                z = np.random.randn(self.horizon, x_reshaped.shape[1], self.nz)
            z = torch.FloatTensor(z).to(x_reshaped.device)
        return self.decode(x_reshaped, z)
"""
