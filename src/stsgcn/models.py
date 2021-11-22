import torch
import torch.nn as nn
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
