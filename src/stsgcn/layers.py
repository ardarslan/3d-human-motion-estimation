import torch
import torch.nn as nn
import math

class SPL(nn.Module):
    def __init__(self, cfg, input_features, hidden_features, output_joints, dense):
        super(SPL, self).__init__()
        
        SMPL_SKELETON = [[(-1, 0, "l_hip"), (-1, 1, "r_hip"), (-1, 2, "spine1")],
                         [(0, 3, "l_knee"), (1, 4, "r_knee"), (2, 5, "spine2")],
                         [(5, 6, "spine3")],
                         [(6, 7, "neck"), (6, 8, "l_collar"), (6, 9, "r_collar")],
                         [(7, 10, "head"), (8, 11, "l_shoulder"), (9, 12, "r_shoulder")],
                         [(11, 13, "l_elbow"), (12, 14, "r_elbow")]]
        H36M_SKELETON = [[(-1, 0, "Hips")],
                         [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
                         [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
                         [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"), (10, 13, "LeftShoulder"), (10, 11, "Neck")],
                         [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"), (11, 12, "Head")],
                         [(18, 19, "RightForeArm"), (14, 15, "LeftForeArm")],
                         [(19, 20, "RightHand"), (15, 16, "LeftHand")]]
        
        self.cfg = cfg
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.dense = True if dense>0 else False
        
        # Unfortunately uses 21 for H3.6M and 15 for SMPL. Dunno which is which, how to port to 22/18
        self.output_joints = output_joints
        self.skeleton = SMPL_SKELETON if "amass" in cfg["dataset"] else H36M_SKELETON
        
        kinematic_tree = dict()
        for joint_list in self.skeleton:
            for joint_entry in joint_list:
                parent_list_ = [joint_entry[0]] if joint_entry[0] > -1 else []
                kinematic_tree[joint_entry[1]] = [parent_list_, joint_entry[1], joint_entry[2]]

        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)
    
        self.prediction_order = list()
        self.indexed_skeleton = dict()
        
        # Reorder the structure so that we can access joint information by using its index.
        self.prediction_order = list(range(len(kinematic_tree)))
        for joint_id in self.prediction_order:
            joint_entry = kinematic_tree[joint_id]
            if not self.dense:
                new_entry = joint_entry
            else:
                parent_list_ = list()
                if len(joint_entry[0]) > 0:
                    get_all_parents(parent_list_, joint_entry[0][0], kinematic_tree)
                new_entry = [parent_list_, joint_entry[1], joint_entry[2]]
            self.indexed_skeleton[joint_id] = new_entry
        
        self.layers = []
        for idx, joint_key in enumerate(self.prediction_order):
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]
            self.layers.append(nn.Sequential([nn.Linear(self.input_features+len(parent_joint_ids)*3, self.hidden_features),
                                              nn.ReLU(), nn.Linear(self.hidden_features, 3)]))    
        
    def forward(self, x):
        joint_predictions = dict()
        for idx, joint_key in enumerate(self.prediction_order):
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]
            joint_inputs = [x]
            for parent_joint_id in parent_joint_ids:
                joint_inputs.append(joint_predictions[parent_joint_id])
            joint_predictions[joint_id] = self.layers[idx](torch.cat(joint_inputs, dim=-1))
            
        return torch.cat(list(joint_predictions.values()), axis=-1)

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
