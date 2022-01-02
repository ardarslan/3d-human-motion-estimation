#Graph.py
#GRAPHS
#Graph J, graph between multiple joints
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def Graph_J(layout, dilation = 1):
    max_hop = 1
    
    ####################Getting the edges
    #for h3.6m dataset
    if layout == 'h36m':
        num_node = 20 #20 joints         
        neighbor_link_ = [(1,2),(2,3),(3,4),(5,6),(6,7),(7,8), (1,9),(5,9),
                            (9,10),(10,11),(11,12),(10,13),(13,14), (14,15),
                            (15,16),(10,17),(17,18),(18,19),(19,20)]    
    #for cmu mocap dataset
    else if layout == 'cmu':
        num_node = 26 #26 joints
        neighbor_link_ = [(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(1,9),(5,9),
                          (9,10),(10,11),(11,12),(12,13),(13,14),
                          (11,15),(15,16),(16,17),(17,18),(18,19),(17,20),
                          (12,21),(21,22),(22,23),(23,24),(24,25),(23,26)]        
    
    #linking neighbouring joints
    link = [(i, i) for i in range(num_node)]
    neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
    #getting the edge
    edge = link + neighbor_link
    center = 10-1

    ####################Getting the adjacency
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    
    #hop_dis = get_hop_distance(num_node, edge)
    
    for hop in valid_hop:
        #getting the hop distance
        hop_dis = get_hop_distance(num_node, edge)
        adjacency[hop_dis == hop] = 1
        
    #normalizing the graphs from adjacency
    normalize_adjacency = normalize_digraph(adjacency)
    A_matrix = np.zeros((1, num_node, num_node))
    A_matrix[0] = normalize_adjacency
    
    return A_matrix


#Graph P, graph between multiple parts
def Graph_P(layout, dilation = 1):
    max_hop = 1
    
    ####################Getting the edges
    #for h3.6m dataset
    if layout == 'h36m'or layout == 'cmu':
        num_node = 10 #10 parts         
        neighbor_link_ = [(1,2),(3,4),(1,5),(3,5),(5,6),(5,7),(7,8),(5,9),(9,10)]
    
    #linking neighbouring parts
    link = [(i, i) for i in range(num_node)]
    neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
    #getting the edge
    edge = link + neighbor_link
    center = 5-1

    ####################Getting the adjacency
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    
    #hop_dis = get_hop_distance(num_node, edge)
   
    for hop in valid_hop:
        #getting the hop distance
        hop_dis = get_hop_distance(num_node, edge)
        adjacency[hop_dis == hop] = 1
        
    #normalizing the graphs from adjacency
    normalize_adjacency = normalize_digraph(adjacency)
    A_matrix = np.zeros((1, num_node, num_node))
    A_matrix[0] = normalize_adjacency
    
    return A_matrix


#Graph B, graph between multiple bodies
def Graph_B(layout, dilation = 1):
    max_hop = 1
    
    ####################Getting the edges
    #for h3.6m dataset
    if layout == 'h36m'or layout == 'cmu':
        num_node = 5 #5 parts         
        neighbor_link_ = [(1,3),(2,3),(3,4),(3,5)]
    
    #linking neighbouring parts
    link = [(i, i) for i in range(num_node)]
    neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
    #getting the edge
    edge = link + neighbor_link
    center = 3-1

    ####################Getting the adjacency
    valid_hop = range(0, max_hop + 1, dilation)
    adjacency = np.zeros((num_node, num_node))
    
    #hop_dis = get_hop_distance(num_node, edge)
   
    for hop in valid_hop:
        #getting the hop distance
        hop_dis = get_hop_distance(num_node, edge)
        adjacency[hop_dis == hop] = 1
        
    #normalizing the graphs from adjacency
    normalize_adjacency = normalize_digraph(adjacency)
    A_matrix = np.zeros((1, num_node, num_node))
    A_matrix[0] = normalize_adjacency
    
    return A_matrix
            
            
#calculating max hop distance  
def get_hop_distance(num_node, edge, max_hop = 1):
    A = np.zeros((num_node, num_node))  
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
        
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


#normalizing the graphs
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
    
##################################
#Operation.py
#Attn, Part and Body Info

#Nodes to Edge Converters
def node2edge(x, rel_rec, rel_send):
    receivers = torch.matmul(rel_rec, x)
    senders = torch.matmul(rel_send, x)
    distance = receivers - senders
    edges = torch.cat([receivers, distance], dim=2)
    return edges

#Edges to Nodes by calculating mean
def edge2node_mean(x, rel_rec, rel_send):
    incoming = torch.matmul(rel_rec.t(), x)
    nodes = incoming/incoming.size(1)
    return nodes

#Main MLP Layers
class Mlp_JpTrans(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.5, out_act=True):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid+n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()
        self.out_act = out_act

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x):
        x_skip = x
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(torch.cat((x,x_skip),-1))
        x = self.batch_norm(x)
        x = self.leaky_relu(x) if self.out_act==True else x
        return x
    
    
#Part Local Information    
class PartLocalInform(nn.Module):

    def __init__(self, layout):
        super().__init__()
        if layout == 'h36m':
            self.torso = [8,9]
            self.left_leg_up = [0,1]
            self.left_leg_down = [2,3]
            self.right_leg_up = [4,5]
            self.right_leg_down = [6,7]
            self.head = [10,11]
            self.left_arm_up = [12,13]
            self.left_arm_down = [14,15]
            self.right_arm_up = [16,17]
            self.right_arm_down = [18,19]
            
        else if layout == 'cmu':
            self.torso = [8,9,10]
            self.left_leg_up = [0,1]
            self.left_leg_down = [2,3]
            self.right_leg_up = [4,5]
            self.right_leg_down = [6,7]
            self.head = [11,12,13]
            self.left_arm_up = [14,15]
            self.left_arm_down = [16,17,18,19]
            self.right_arm_up = [20,21]
            self.right_arm_down = [22,23,24,25]

    def forward(self, part, layout):
        N, d, T, w = part.size()  # [64, 256, 7, 10]
        
        if layout == 'h36m':
            x = part.new_zeros((N, d, T, 20))
        else if layout == 'cmu':
            x = part.new_zeros((N, d, T, 26))

        x[:,:,:,self.left_leg_up] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = torch.cat((part[:,:,:,2].unsqueeze(-1), part[:,:,:,2].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,6].unsqueeze(-1),part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = torch.cat((part[:,:,:,7].unsqueeze(-1),part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,8].unsqueeze(-1),part[:,:,:,8].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = torch.cat((part[:,:,:,9].unsqueeze(-1),part[:,:,:,9].unsqueeze(-1)),-1)

        return x

#Body Local Information
class BodyLocalInform(nn.Module):

    def __init__(self, layout):
        super().__init__()
        
        if layout == 'h36m':
            self.torso = [8,9,10,11]
            self.left_leg = [0,1,2,3]
            self.right_leg = [4,5,6,7]
            self.left_arm = [12,13,14,15]
            self.right_arm = [16,17,18,19]
        
        else if layout == 'cmu':
            self.torso = [8,9,10,11,12,13]
            self.left_leg = [0,1,2,3]
            self.right_leg = [4,5,6,7]
            self.left_arm = [14,15,16,17,18,19]
            self.right_arm = [20,21,22,23,24,25]

    def forward(self, body, layout):
        N, d, T, w = body.size()  # [64, 256, 7, 10]
        
        if layout == 'h36m':
            x = part.new_zeros((N, d, T, 20))
        else if layout == 'cmu':
            x = part.new_zeros((N, d, T, 26))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,2:3]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x
    

#Attention Information (joint, part and body, just use the same for everything)
class AttInform(nn.Module):

    def __init__(self, n_1, n_2, t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=False):
        super().__init__()
        
        self.time_conv = nn.Sequential(nn.Conv2d(n_1, n_1, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_1),
                                                 nn.Dropout(drop, inplace=True))
        if nmp==True:
            self.mlp1 = Mlp_JpTrans(n_2[0], n_2[1], n_2[1], drop)
            self.mlp2 = Mlp_JpTrans(n_2[1]*2, n_2[1], n_2[1], drop)
            self.mlp3 = Mlp_JpTrans(n_2[1]*2, n_2[1], n_2[1], drop, out_act=False)
        else:
            self.mlp1 = Mlp_JpTrans(n_2[0], n_2[1], n_2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send):                                           # x: [64, 32, 49, 10]
        N, D, T, V = x.size()
        x_ = x if self.layer1==True else self.time_conv(x)
        x_ = x_.permute(0,2,3,1)
        x_ = x_.contiguous().view(N,V,-1)
        x_node = self.mlp1(x_)
        if self.nmp==True:
            x_node_skip = x_node
            x_edge = node2edge(x_node, rel_rec, rel_send)                               # [64, 420, 512]
            x_edge = self.mlp2(x_edge)                                                  # [64, 420, 256]
            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                          # [64, 21, 256]
            x_node = torch.cat((x_node, x_node_skip), -1)                               # [64, 21, 512]
            x_node = self.mlp3(x_node)                                                  # [64, 21, 256]
        return x_node


#Module.py
#Stgcn, SpatialConv, DecodeGCN, Average Joint and Average Part, S1-S2-S3
########################
#########################
##########################

#temporal and graph convolutional network
class St_gcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, stride=1, dropout=0.5, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0]-1)//2, 0)

        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A_skl):
        res = self.residual(x)
        x = self.gcn(x, A_skl)
        x = self.tcn(x) + res
        return self.relu(x)


class SpatialConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, 
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*(k_num),
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A_skl):
        x = self.conv(x)                                               # [64, 128, 49, 21]
        n, kc, t, v = x.size()                                         # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num,  kc//(self.k_num), t, v)             # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))
        return x.contiguous()

#decode GCN
class DecodeGcn(nn.Module):
    
    def __init__(self, in_channels, out_channels, k_num,
                 kernel_size=1, stride=1, padding=0,
                 dilation=1, dropout=0.5, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels*(k_num), 
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_skl):      # x: [64, 256, 21] = N, d, V
        x = self.conv(x)
        x = self.dropout(x)
        n, kc, v = x.size()
        x = x.view(n, (self.k_num), kc//(self.k_num), v)          # [64, 4, 256, 21]
        x = torch.einsum('nkcv,kvw->ncw', (x, A_skl))           # [64, 256, 21]
        return x.contiguous()


#find the average joint pos for part
class AverageJoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9]
        self.left_leg_up = [0,1]
        self.left_leg_down = [2,3]
        self.right_leg_up = [4,5]
        self.right_leg_down = [6,7]
        self.head = [10,11]
        self.left_arm_up = [12,13]
        self.left_arm_down = [14,15]
        self.right_arm_up = [16,17]
        self.right_arm_down = [18,19]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 2))                                   # [N, C, T, V=1]
        x_leftlegup = F.avg_pool2d(x[:, :, :, self.left_leg_up], kernel_size=(1, 2))                         # [N, C, T, V=1]
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                     # [N, C, T, V=1]
        x_rightlegup = F.avg_pool2d(x[:, :, :, self.right_leg_up], kernel_size=(1, 2))                       # [N, C, T, V=1]
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                   # [N, C, T, V=1]
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 2))                                     # [N, C, T, V=1]
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                         # [N, C, T, V=1]
        x_leftarmdown = F.avg_pool2d(x[:, :, :, self.left_arm_down], kernel_size=(1, 2))                     # [N, C, T, V=1]
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                       # [N, C, T, V=1]
        x_rightarmdown = F.avg_pool2d(x[:, :, :, self.right_arm_down], kernel_size=(1, 2))                   # [N, C, T, V=1]
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head,  x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)               # [N, C, T, V=1]), dim=-1)        # [N, C, T, 10]
        return x_part


#find the average part pose for body
class AveragePart(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9,10,11]
        self.left_leg = [0,1,2,3]
        self.right_leg = [4,5,6,7]
        self.left_arm = [12,13,14,15]
        self.right_arm = [16,17,18,19]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 4))                                 # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 4))                            # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 4))                          # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 4))                            # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 4))                          # [N, C, T, V=1]
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)                # [N, C, T, V=1]), dim=-1)        # [N, C, T, 10]
        return x_body
    
#Transforms going from joints/parts to parts/body respectively
class S2S_forward(nn.Module):

    def __init__(self, n_init1, n_init2, n_fin1, n_fin2, t_kernel, t_stride, t_padding):
        super().__init__()
        self.embed_s1 = AttInform(n_init1, n_init2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s2 = AttInform(n_fin1, n_fin2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s1, x_s2, relrec_s1, relsend_s1, relrec_s2, relsend_s2):                                                           # x: [64, 3, 49, 21]
        N, d, T, V = x_s1.size()
        N, d, T, W = x_s2.size()

        x_s1_att = self.embed_s1(x_s1, relrec_s1, relsend_s1)                                                                          # [64, 21, 784]
        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2)
        Att = self.softmax(torch.matmul(x_s1_att, x_s2_att.permute(0,2,1)).permute(0,2,1))       # [64, 10, 21]

        x_s1 = x_s1.permute(0,3,2,1).contiguous().view(N,V,-1)              # [64, 21, 49, 3] -> [64, 21, 147]
        x_s2_glb = torch.einsum('nwv, nvd->nwd', (Att, x_s1))               # [64, 10, 147]

        # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]
        x_s2_glb = x_s2_glb.contiguous().view(N, W, -1, d).permute(0,3,2,1) 
        
        return x_s2_glb
    

#Transforms going from parts/body to joints/parts respectively
class S2S_backward(nn.Module):

    def __init__(self, n_fin1, n_fin2, n_init1, n_init2, t_kernel, t_stride, t_padding):
        super().__init__()
        self.embed_s2 = AttInform(n_fin1, n_fin2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.embed_s1 = AttInform(n_init1, n_init2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_s2, x_s1, relrec_s1, relsend_s1, relrec_s2, relsend_s2):                                                           # x: [64, 3, 49, 21]
        N, d, T, W = x_s2.size()
        N, d, T, V = x_s1.size()

        x_s2_att = self.embed_s1(x_s2, relrec_s2, relsend_s2)                                                                          # [64, 21, 784]
        x_s1_att = self.embed_s2(x_s1, relrec_s1, relsend_s1)
        Att = self.softmax(torch.matmul(x_s2_att, x_s1_att.permute(0,2,1)).permute(0,2,1))       # [64, 10, 21]

        x_s2 = x_s2.permute(0,3,2,1).contiguous().view(N,W,-1)              # [64, 21, 49, 3] -> [64, 21, 147]
        x_s1_glb = torch.einsum('nvw, nwd->nvd', (Att, x_s1))               # [64, 10, 147]

        # [64, 10, 784] -> [64, 10, 49, 16] -> [64, 3, 49, 10]
        x_s1_glb = x_s1_glb.contiguous().view(N, W, -1, d).permute(0,3,2,1) 
        
        return x_s1_glb