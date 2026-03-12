import torch
import dgl
from torch import nn, Tensor
from collections.abc import Mapping
from dgl import backend as F
from collections import defaultdict
from dgl import function as fn
from typing import Optional
from models.Attention import MultiHeadCrossAttention, SiLUMLP, process_mask

# class NeighborSampler(object):
#     def __init__(self, g, num_hops, device=None):
#         """
#         g 为 DGLGraph；
#         fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
#         """
#         self.g = g
#         self.num_hops = num_hops
#         self.device = device


#     def sample_blocks(self, seeds:torch.Tensor):
#         blocks = []
#         seeds = seeds.clone().detach().to(torch.int64)
#         seeds =  {'cell':seeds}
#         for cnt in range(self.num_hops): 
#             frontier = dgl.in_subgraph(self.g,seeds)
#             block = dgl.to_block(frontier, seeds)
#             seeds = block.srcdata[dgl.NID]['aig']
#             seeds = {'aig':seeds}
#             blocks.insert(0, block)
#         return blocks
    
# With Cell2Cell Edge
class NeighborSampler(object):
    def __init__(self, g, num_hops, device=None):
        """
        g 为 DGLGraph；
        fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
        """
        self.g = g
        self.num_hops = num_hops
        self.device = device


    def sample_blocks(self, seeds:torch.Tensor):
        blocks = []
        seeds = seeds.clone().detach().to(torch.int64)
        seeds =  {'cell':seeds}
        for cnt in range(self.num_hops): 
            frontier = dgl.in_subgraph(self.g,seeds)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        frontier = dgl.in_subgraph(self.g,{'cell':seeds['cell']})
        sub_g = frontier.edge_type_subgraph(['mapper'])
        block = dgl.to_block(sub_g, seeds)
        blocks.insert(0, block)
        return blocks

def expand_as_pair(input_, g=None):
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()
            }
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

import torch
import torch.nn as nn

def batch_pearson(x, num_head, learnable_diag):
    """
    输入: x [B, N, D] 特征矩阵
    输出: [B, N, N] 皮尔逊相关系数矩阵，且对角线可学习
    """
    B, N, D = x.shape
    mean_x = x.mean(dim=-1, keepdim=True)  # [B, N, 1]
    x_centered = x - mean_x  # [B, N, D]
    # 计算协方差
    cov = torch.matmul(x_centered, x_centered.transpose(-1, -2))  # [B, N, N]
    # 计算标准差
    std = torch.sqrt(torch.sum(x_centered ** 2, dim=-1, keepdim=True))  # [B, N, 1]
    std_matrix = torch.matmul(std, std.transpose(-1, -2))  # [B, N, N]
    # 计算皮尔逊相关系数
    epsilon = 1e-6
    pearson_corr = cov / (std_matrix + epsilon)  # [B, N, N]
    # 替换对角线为可学习参数
    diag_mask = torch.eye(N, device=x.device).expand(B, num_head, -1, -1)
    pearson_corr = pearson_corr.unsqueeze(1).expand(B, num_head, N, N)
    corr_matrix = pearson_corr * (1 - diag_mask) + diag_mask * learnable_diag.unsqueeze(2)
    corr_matrix = nn.functional.softmax(corr_matrix, dim=-1)

    return corr_matrix

class RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))
  
  def _norm(self, hidden_states: Tensor) -> Tensor:
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + self.eps)
  
  def forward(self, hidden_states: Tensor) -> Tensor:
    return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
    
# class RMSNorm(nn.Module):
#     def __init__(self, d_model, eps=1e-8):
#         """
#         d_model: 输入特征的维度
#         eps: 防止除零的微小数值，确保数值稳定性
#         """
#         super().__init__()
#         self.eps = eps
#         # 可学习的缩放参数，初始化为1
#         self.scale = nn.Parameter(torch.ones(d_model))
#         # 可选：可学习的偏移参数，初始化为0
#         self.bias = nn.Parameter(torch.zeros(d_model))

#     def forward(self, x):
#         # 计算RMS (均方根)，沿着最后一个维度进行
#         rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
#         # 对输入进行RMS归一化
#         x_norm = x / rms
#         # 应用可学习的缩放参数和偏移
#         return self.scale * x_norm + self.bias
    
# class LammaMLPv2(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: int,
#         multiple_of: int,
#         ffn_dim_multiplier: Optional[float] = None,
#     ):
#         """
#         Initialize the FeedForward module.

#         Args:
#             dim (int): Input dimension.
#             hidden_dim (int): Hidden dimension of the feedforward layer.
#             multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
#             ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
#         """
#         super().__init__()
#         hidden_dim = int(2 * hidden_dim / 3)
#         if ffn_dim_multiplier is not None:
#             hidden_dim = int(ffn_dim_multiplier * hidden_dim)
#         hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

#         # 定义三个线性层
#         self.w1 = nn.Linear(dim, hidden_dim, bias=False)
#         self.w2 = nn.Linear(hidden_dim, dim, bias=False)
#         self.w3 = nn.Linear(dim, hidden_dim, bias=False)

#     def forward(self, x):
#         """
#         Forward pass of the FeedForward module.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_length, dim).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, seq_length, dim).
#         """
#         return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
 

class AttnSageConv(nn.Module):
    def __init__(self, in_feat, hidden_dim, out_feat, hop, lamma_multi = 8, num_heads=4, mlp_hidden_dim=32, drop_rate=0.1, gamma=0.7, slope=0.1, bias=True, pearson=None, device=None):
        super(AttnSageConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feat)
        self._out_feats = out_feat
        self.pre_norm = RMSNorm(in_feat)
        self.after_norm = RMSNorm(out_feat)
        # self.pre_norm = nn.LayerNorm(in_feat)
        # self.after_norm = nn.LayerNorm(out_feat)

        self.pre_projection_cell = nn.Linear(in_feat, hidden_dim)
        self.pre_projection_and = nn.Linear(in_feat, hidden_dim, bias=False)
        self.feat_drop = nn.Dropout(drop_rate)
        self.activation = nn.SiLU()
        self.neigh_fc = nn.Linear(hidden_dim, out_feat, bias)
        self.self_fc = nn.Linear(hidden_dim, out_feat, bias)
        self.num_heads = num_heads
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # cell agg
        self.gamma = gamma
        self.slope = slope
        self.hop = hop
        self.pearson = pearson
        self.cell_attn = MultiHeadCrossAttention(self.num_heads, hidden_dim, drop_rate)
        self.cell2cell_attn = MultiHeadCrossAttention(self.num_heads, hidden_dim, drop_rate)
        self.and_attn = MultiHeadCrossAttention(self.num_heads, hidden_dim, drop_rate)
        self.not_attn = MultiHeadCrossAttention(self.num_heads, hidden_dim, drop_rate)

        self.mlp = SiLUMLP(out_feat, mlp_hidden_dim, out_feat, 3, drop_rate)
        self.cross_fc = SiLUMLP(hidden_dim*2,mlp_hidden_dim,hidden_dim)
        self.reset_param()
    
    def reset_param(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.kaiming_uniform_(self.pre_projection_cell.weight) 
        nn.init.kaiming_uniform_(self.pre_projection_and.weight) 
        for linear in self.cell_attn.linears:
            nn.init.kaiming_uniform_(linear.weight) 
        for linear in self.and_attn.linears:
            nn.init.kaiming_uniform_(linear.weight) 
        for linear in self.not_attn.linears:
            nn.init.kaiming_uniform_(linear.weight) 
        nn.init.kaiming_uniform_(self.neigh_fc.weight) 

    def cross_reducer(self, flist):
        if len(flist) > 1:
            feat_comb = torch.concat(flist, dim=-1)
            feat = self.cross_fc(feat_comb)
        else:
            feat = flist[0]
        return feat

    def self_attn_agg(self, nodes):
        find_ids = nodes.nodes()
        # print(nodes.data)
        gate_type = nodes.data['gate_type']
        # print(gate_type)
        and_ind = (gate_type == 1).nonzero(as_tuple=True)[0]
        not_ind = (gate_type == 2).nonzero(as_tuple=True)[0]
        indices = torch.nonzero(self.ori_ids['aig'].unsqueeze(0) == find_ids.unsqueeze(1), as_tuple=True)[1]
        m_self = self.hf['aig'][indices] # B 1 D
        m_neigh = nodes.mailbox['m_a'] # B N D
        m_self=m_self.unsqueeze(1)
        out_rst = torch.zeros_like(m_self, device=self.device)
        
        # print(m_self.shape, m_neigh.shape)
        # Agg And Gate
        and_self = torch.index_select(m_self, dim=0, index=and_ind)
        and_neigh = torch.index_select(m_neigh, dim=0, index=and_ind)
        # print(and_self.shape, and_neigh.shape)
        agg_feat = self.and_attn(and_self, and_neigh, and_neigh)
        out_rst[and_ind] = agg_feat

        # Agg Not Gate
        not_self = torch.index_select(m_self, dim=0, index=not_ind)
        not_neigh = torch.index_select(m_neigh, dim=0, index=not_ind)
        agg_feat = self.not_attn(not_self, not_neigh, not_neigh)
        out_rst[not_ind] = agg_feat
        # print(out_rst.shape, m_self.shape)

        # out_put = self.activation(self.neigh_fc(agg_feat))
        return {'neigh': out_rst+m_self}
    
    def cell_msg_func(self, edges):
        return {'m_c':edges.src['hf'], 'tm_c':edges.src['topo']}

    def cell_attn_agg(self, nodes):
        m_neigh = nodes.mailbox['m_c'] # B,N,D
        m_neigh_topo = nodes.mailbox['tm_c'] # B,N,1
        b,_,_ = m_neigh_topo.shape
        find_ids = nodes.nodes()
        indices = torch.nonzero(self.ori_ids['cell'].unsqueeze(0) == find_ids.unsqueeze(1), as_tuple=True)[1]
        m_self = self.hf['cell'][indices] # B 1 D
        m_self = m_self.unsqueeze(1)

        # print('cell_fanin,root: ', m_self.shape, m_neigh.shape)
        max_topo = torch.max(m_neigh_topo,dim=1).values + 1
        max_topo = max_topo.unsqueeze(1)
        relative_topo_d = max_topo - m_neigh_topo
        relative_topo_d = relative_topo_d.squeeze(-1)

        m_neigh = torch.concat([m_self, m_neigh], dim=1)
        relative_topo_d = torch.cat([torch.zeros((b,1)).to(self.device), relative_topo_d], dim=1)

        mask_head = process_mask(relative_topo_d,self.gamma,self.hop,self.slope)
        pearson_matrix=None
        # mask_head = None
        if self.pearson is not None:
            pearson_matrix = batch_pearson(m_neigh, self.num_heads, self.pearson)
        # print('rel_topo_d ', relative_topo_d.shape, relative_topo_d)

        # with open('./file_test.log', 'a') as f:
        #     f.write('relative_topo_d\n')
        #     for i in range(relative_topo_d.shape[0]):
        #         f.write(str(relative_topo_d[i].tolist()) + '\n')    
        #         f.write(str(mask_head[i].tolist()) + '\n')    
        #     f.write('attn_weight\n')
        agg_feat = self.cell_attn(m_self, m_neigh, m_neigh, mask_head, pearson_matrix, printout=True)
        # agg_feat = self.activation(self.neigh_fc(agg_feat))
        return {'neigh':agg_feat+m_self}
    
    def cell2cell_agg(self, nodes):
        m_neigh = nodes.mailbox['m_cc'] # B,N,D
        b,_,_ = m_neigh.shape
        find_ids = nodes.nodes()
        indices = torch.nonzero(self.ori_ids['cell'].unsqueeze(0) == find_ids.unsqueeze(1), as_tuple=True)[1]
        m_self = self.hf['cell'][indices] # B 1 D
        m_self = m_self.unsqueeze(1)

        m_neigh = torch.concat([m_self, m_neigh], dim=1)
        # print('rel_topo_d ', relative_topo_d.shape, relative_topo_d)

        agg_feat = self.cell2cell_attn(m_self, m_neigh, m_neigh)
        # agg_feat = self.activation(self.neigh_fc(agg_feat))
        return {'neigh':agg_feat+m_self}
    
    def post_agg(self, ori_feat, h_self, h_neigh):
        rst = self.activation(self.self_fc(h_self) + self.neigh_fc(h_neigh))
        out_feat = self.after_norm(rst)
        out_feat = self.activation(self.mlp(out_feat))

        out_feat = out_feat + rst # skip connection
        # out_feat = out_feat + ori_feat
        return out_feat

    
    def forward(self, graph:dgl.DGLGraph, feat, emb_src=None):
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = feat[0]
            feat_dst = feat[1]
            # print(feat_src)
            # print(feat_dst)
            # print(feat_src['aig'].shape)
            # print(feat_src['cell'].shape)
            # print(feat_dst['aig'].shape)
            # print(feat_dst['cell'].shape)
            # torch.save(feat_dst['cell'], './temp_tensor.pt')
            feat_aig_src = self.pre_projection_and(self.pre_norm(self.feat_drop(feat_src['aig'])))
            feat_cell_src = self.pre_projection_cell(self.pre_norm(self.feat_drop(feat_src['cell'])))
            feat_aig_dst = self.pre_projection_and(self.pre_norm(self.feat_drop(feat_dst['aig'])))
            feat_cell_dst = self.pre_projection_cell(self.pre_norm(self.feat_drop(feat_dst['cell'])))
        else:
            feat_src = feat_dst = self.feat_drop(feat)
        
        if emb_src:
            emb_aig_src = emb_src['aig']
            emb_cell_src = emb_src['cell']
        else:
            emb_aig_src = graph.srcdata['topo_feat']['aig']
            emb_cell_src = graph.srcdata['topo_feat']['cell']
            emb_aig_dst = graph.dstdata['topo_feat']['aig']
            emb_cell_dst = graph.dstdata['topo_feat']['cell']

        h_self = {'aig':feat_aig_dst, 'cell':feat_cell_dst}

        # print(graph, feat_aig_dst.shape, feat_cell_dst.shape)
        self.hf = {'aig':feat_aig_src, 'cell':feat_cell_src}
        graph.ndata['hf'] = {'aig':feat_aig_src, 'cell':feat_cell_src}
        graph.ndata['topo'] = {'aig':emb_aig_src, 'cell':emb_cell_src}
        self.ori_ids = graph.srcdata[dgl.NID]

        funcs = defaultdict()
        funcs[('aig','and','aig')] = (fn.copy_u('hf','m_a'),self.self_attn_agg)
        funcs[('aig','mapper','cell')] = (self.cell_msg_func,self.cell_attn_agg)
        funcs[('cell','cell_edge','cell')] = (fn.copy_u('hf','m_cc'),self.cell2cell_agg)
        # print(graph)
        graph.multi_update_all(funcs,self.cross_reducer)

        # feat_src = self.pre_norm(feat_src)
        # graph.srcdata['h'] = feat_src
        # # print("src_dst_shape: ", feat_src.shape, feat_dst.shape)
        # graph.update_all(fn.copy_u('h_a','m_a'), self.self_attn_agg)
        h_neigh = graph.dstdata['neigh']
        aig_feat = None
        if 'aig' in h_neigh:
            aig_feat = self.post_agg(self.pre_norm(self.feat_drop(feat_dst['aig'])), h_self['aig'], h_neigh['aig'].squeeze())
        cell_feat = self.post_agg(self.pre_norm(self.feat_drop(feat_dst['cell'])), h_self['cell'], h_neigh['cell'].squeeze())

        return {'aig':aig_feat, 'cell':cell_feat}

class AttnGraphSage(nn.Module):
    def __init__(self, args, hop, pearson, device=None):
        super(AttnGraphSage, self).__init__()
        self.n_layer = args.n_layer
        self.layers = nn.ModuleList()
        n_hidden = args.hidden_dim
        in_feat = args.in_feat
        self.pearson = pearson
        out_feat = args.out_feat 
        lamma2_multi = args.lamma2_multi
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.layers.append(AttnSageConv(in_feat,n_hidden,out_feat, hop, lamma2_multi, args.num_heads, args.mlp_hidden_dim, args.drop_rate, args.gamma, args.slope, args.bias, self.pearson, self.device))
        for i in range(1,self.n_layer-1):
            self.layers.append(AttnSageConv(out_feat,n_hidden,out_feat, hop, lamma2_multi, args.num_heads, args.mlp_hidden_dim, args.drop_rate, args.gamma, args.slope, args.bias, self.pearson, self.device))
        self.layers.append(AttnSageConv(out_feat,n_hidden,out_feat, hop, lamma2_multi, args.num_heads, args.mlp_hidden_dim, args.drop_rate, args.gamma, args.slope, args.bias, self.pearson, self.device))
        self.drop = nn.Dropout(args.drop_rate)
        self.activation = nn.SiLU()
    
    def forward(self, blocks, x_h):
        """
        @param blocks hetergeneous block
        @param x_h dict{'aig':hf, 'cell':hf}
        @param x_f dict{'aig':topo_feat, 'cell':topo_feat}
        """
        h = x_h
        blocks = blocks[1:]
        for l, (layer,block) in enumerate(zip(self.layers,blocks)):
            h_src = {'aig':h['aig'][:block.number_of_src_nodes(ntype='aig')], 'cell':h['cell'][:block.number_of_src_nodes(ntype='cell')]}
            h_dst = {'aig':h['aig'][:block.number_of_dst_nodes(ntype='aig')], 'cell':h['cell'][:block.number_of_dst_nodes(ntype='cell')]}
            h_rst = layer(block,(h_src,h_dst))
            if l != len(self.layers) - 1:
                for n_type in h_rst.keys():
                    if h_rst[n_type] is not None:
                        h_rst[n_type] = self.activation(h_rst[n_type])
                        h_rst[n_type] = self.drop(h_rst[n_type])
            if h_rst['aig'] is not None:
                h['aig'][:block.number_of_dst_nodes(ntype='aig')] = h_rst['aig']
            h['cell'][:block.number_of_dst_nodes(ntype='cell')] = h_rst['cell']
        return h['cell'][:blocks[-1].number_of_dst_nodes(ntype='cell')]
