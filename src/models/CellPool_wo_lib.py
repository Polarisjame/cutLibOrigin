import torch
from torch import nn
import deepgate
import os
import dgl
import json
import time
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset 

from models.attn_sage import AttnGraphSage, SiLUMLP
from models.Attention import PointerNetwork


class CellPool(nn.Module):
    def __init__(self, args, device=None, activate = None):
        super(CellPool, self).__init__()

        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
        else:
            self.device = device
        self.data_root = args.data_root

        self.deepgate = deepgate.Model()
        self.deepgate.load_pretrained(args.dg_pretrained_path) 
        # self.deepgate.train()

        self.hop = nn.Parameter(torch.full((args.num_heads,1),float(args.num_hop)))
        # self.pearson = nn.Parameter(torch.full((args.num_heads,1),float(args.diag)))
        self.pearson = None
        self.fanout_rate = nn.Parameter(torch.full((1,1),float(args.fanout_rate)))
        self.cell_fanout_embedding = nn.Embedding(10, args.in_feat)
        self.fanout_bins = torch.tensor([10, 50, 100, 150, 200, 300, 500, 1500, 3000]).to(self.device)
        self.fanout_projection = SiLUMLP(args.in_feat*2, args.mlp_hidden_dim, args.in_feat)
        # self.liberty_proj = SiLUMLP(49*4, args.mlp_hidden_dim, args.in_feat)

        # self.fanout_projection = SiLUMLP(1,128,args.in_feat)
        self.gnnmodel = AttnGraphSage(args, self.hop, self.pearson, device=self.device)
        self.activate=None
        self.feat_drop = nn.Dropout(args.drop_rate)
        if activate == 'relu':
            self.activate=nn.ReLU()
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()
        elif activate == 'prelu':
            self.activate = nn.PReLU()

        self.liberty_comb = []
        self.liberty_comb.append(SiLUMLP(args.in_feat*2, args.mlp_hidden_dim, args.in_feat))
        self.liberty_comb.append(self.activate)
        self.liberty_comb.append(nn.LayerNorm(args.in_feat))
        self.liberty_comb.append(self.feat_drop)
        self.liberty_comb = torch.nn.Sequential(*self.liberty_comb)

        self.projection = []
        # self.projection.append(SiLUMLP(args.out_feat, args.mlp_hidden_dim, args.out_feat))
        # self.projection.append(self.activate)
        # self.projection.append(self.feat_drop)
        self.projection.append(nn.Linear(args.out_feat,1))
        self.projection.append(self.activate)
        self.projection = torch.nn.Sequential(*self.projection)

    # def dg_embedding(self, dg_graph, block_hs, block_hf):
    #     hs, hf = self.deepgate(dg_graph, block_hs, block_hf) 
    #     hs = self.feat_drop(hs)
    #     hf = self.feat_drop(hf)
    #     return hs, hf
        
    
    # def dg_subgraph(self, graphset, blocks):
    #     start = time.time()
    #     dg_graph = OrderedDict()
    #     src_aig_node = blocks[0].srcdata[dgl.NID]['aig']
    #     subg = dgl.node_subgraph(graphset['graph'], {'aig': src_aig_node,'cell':torch.zeros(0,dtype=torch.int64)})
    #     gate_type = subg.ndata['gate_type']['aig']
    #     subg = dgl.to_homogeneous(subg)
    #     x_data = []
    #     for node in subg.nodes():
    #         gate = gate_type[node]
    #         x_data.append([int(node.data), int(gate.data)])
    #     edge_index = subg.edges()
    #     dg_graph = parse_aig2dg(np.array(x_data), np.array(list(torch.stack(edge_index,dim=0).transpose(0,1))))
    #     eps = graphset['eps']
    #     hf:torch.Tensor = graphset['hf'][src_aig_node+eps['aig']]
    #     hs = graphset['hs'][src_aig_node+eps['aig']]
    #     hf.requires_grad_(True)
    #     hs.requires_grad_(True)
    #     hs.to(self.device)
    #     hf.to(self.device)
    #     dg_graph.to(self.device)
    #     self.deepgate.to(self.device)
    #     hs_c, hf_c = self.deepgate(dg_graph, hs, hf)
    #     check = time.time()
    #     return hs_c, hf_c, check-start

    def cell_pooling(self, blocks, hf, feat_dim, cell_fanout, neighbor_dict):
        h_cell = torch.zeros((blocks[0].number_of_nodes(ntype='cell'),feat_dim)).to(self.device)
        for cell_id in blocks[0].nodes(ntype='cell'):
            neighbor_id = neighbor_dict[cell_id.item()]
            h_cell[cell_id] = torch.mean(hf[neighbor_id],dim=0)
        # h_cell = torch.zeros((blocks[-1].number_of_nodes(ntype='cell'),feat_dim)).to(self.device)
        # for cell_id in blocks[-1].nodes(ntype='cell'):
        #     neighbor_id = neighbor_dict[cell_id.item()]
        #     h_cell[cell_id] = torch.mean(hf[neighbor_id],dim=0)
        fanout_categories = torch.bucketize(cell_fanout, self.fanout_bins)
        fanout_embedding = self.cell_fanout_embedding(fanout_categories).squeeze(1)
        # print(cell_fanout.shape, h_cell.shape, fanout_embedding.shape)
        h_cell = torch.cat((h_cell, fanout_embedding), dim=-1) # n*2d
        cell_feat = self.liberty_comb(h_cell)

        # h_cell = torch.cat((h_cell, fanout_embedding, liberty_attn), dim=-1) # n*2d
        return cell_feat

    def forward(self, blocks, batch_inputs):
        # batch_feats = {'aig':self.graph.ndata['topo_feat']['aig'][input_nodes_aig].to(self.device),'cell':self.graph.ndata['topo_feat']['cell'][input_nodes_cell].to(self.device)}
        blocks = [block_.to(self.device) for block_ in blocks]
        # print(batch_inputs)
        batch_out = self.gnnmodel(blocks, batch_inputs)
        cell_feat = batch_out

        pred_res = self.projection(cell_feat)
        # if self.activate:
        #     pred_res = self.activate(pred_res)
        
        return pred_res


class Poolset(Dataset):
    def __init__(self, train_nid, labels):
        self.nids = train_nid
        self.labels = labels
    
    def __getitem__(self, index):
        return self.nids[index], self.labels[index]
    
    def __len__(self):
        return len(self.nids)
    
    def get_labels(self):
        return self.labels
        
class ezPool(nn.Module):
    def __init__(self, infeat, hidden, outfeat):
        super(ezPool, self).__init__()
        self.projection = LammaMLPv2(infeat, hidden, outfeat)
        self.active = nn.PReLU()
        self.drop = nn.Dropout(0.2)
    
    def forward(self, hf):
        agg = torch.mean(hf, dim=0)
        out = self.active(self.drop(self.projection(agg)))
        return out