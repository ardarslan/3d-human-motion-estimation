import torch
import torch.nn as nn
import math


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, time_dim, joints_dim):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(
            torch.FloatTensor(time_dim, joints_dim, joints_dim)
        )  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1.0 / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1.0 / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        """
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        """

    def forward(self, x):
        x = torch.einsum("nctv,vtq->ncqv", (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum("nctv,tvw->nctw", (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        time_dim,
        joints_dim,
        dropout,
        bias=True,
    ):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.gcn = ConvTemporalGraphical(time_dim, joints_dim)  # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual = nn.Identity()
        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class CNN_layer(
    nn.Module
):  # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)
    def __init__(self, in_channels, out_channels, kernel_size, dropout, bias=True):
        super(CNN_layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
            (kernel_size[0] - 1) // 2,
            (kernel_size[1] - 1) // 2,
        )  # padding so that both dimensions are maintained
            
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, num_layers):
        super(TransformerDecoderLayer, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

    def forward(self, tgt, memory, tgt_mask):
        # tgt and memory shape: (batch_size, sequence_len, embedding_size):[256,25,66]
        # tgt_mask.shape : [25,25]
        tgt = self.pos_encoder(tgt)
        return self.transformer_decoder(tgt, memory, tgt_mask)


class PositionalEncoding(nn.Module):
    # borrowed from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        self.pe = self.pe.to(x.device)
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)
