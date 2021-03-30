import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class NodeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(NodeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l - 1] if l > 0 else in_features * 2,
                out_channels=num_features_list[l],
                kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            if l < (len(num_features_list) - 1):
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        # get eye matrix (batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2)
        node_feat = node_feat.squeeze(-1).squeeze(0)
        return node_feat


class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l-1] if l > 0 else in_features,
                out_channels=num_features_list[l], kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        # add final similarity kernel
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1],
                                           out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0)
        # normalize affinity matrix
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)
        return edge_feat, sim_val


class ClassifierGNN(nn.Module):
    def __init__(self, in_features, edge_features, nclasses, device):
        super(ClassifierGNN, self).__init__()

        self.edge_net = EdgeNet(in_features=in_features,
                                num_features=edge_features,
                                device=device)
        # set edge to node
        self.node_net = NodeNet(in_features=in_features,
                                num_features=nclasses,
                                device=device)
        # mask value for no-gradient edges
        self.mask_val = -1

    def label2edge(self, targets):
        ''' convert node labels to affinity mask for backprop'''
        num_sample = targets.size()[1]
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float()
        target_edge_mask = (torch.eq(label_i, self.mask_val) + torch.eq(label_j, self.mask_val)).type(torch.bool)
        source_edge_mask = ~target_edge_mask
        init_edge = edge * source_edge_mask.float()
        return init_edge[0], source_edge_mask

    def forward(self, init_node_feat):
        #  compute normalized and not normalized affinity matrix
        edge_feat, edge_sim = self.edge_net(init_node_feat)
        # compute node features and class logits
        logits_gnn = self.node_net(init_node_feat, edge_feat)
        return logits_gnn, edge_sim
