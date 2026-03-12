import os
import numpy as np
from torch.utils.data import DataLoader

from models.attn_sage import NeighborSampler,dgl,torch
from subutils.config import get_opt
from subutils.tools import OrderedData

def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x

def construct_node_feature(x, num_gate_types):
    # the one-hot embedding for the gate types
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    # if node_reconv:
    #     reconv = torch.tensor(x[:, 7], dtype=torch.float).unsqueeze(1)
    #     x_torch = torch.cat([x_torch, reconv], dim=1)
    return x_torch

def top_sort(edge_index, graph_size):
    node_ids = np.arange(graph_size, dtype=int)
    node_order = np.zeros(graph_size, dtype=int)
    unevaluated_nodes = np.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]
        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False
        n += 1

    return torch.from_numpy(node_order).long()

def return_order_info(edge_index, num_nodes):
    ns = torch.LongTensor([i for i in range(num_nodes)])
    try:
        if len(edge_index[0]) == 1:
            forward_level = [0, 0]
            forward_level[edge_index[1][0]] = 1
            forward_level = torch.tensor(forward_level).long()
        else:
            forward_level = top_sort(edge_index, num_nodes)
    except IndexError:
        print(edge_index)
    ei2 = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    if len(ei2[0]) == 1:
        backward_level = [0, 0]
        backward_level[edge_index[0][0]] = 1
        backward_level = torch.tensor(backward_level).long()
    else:
        backward_level = top_sort(ei2, num_nodes)
    forward_index = ns
    backward_index = torch.LongTensor([i for i in range(num_nodes)])
    
    return forward_level, forward_index, backward_level, backward_index

def parse_aig2dg(x_data:np.ndarray, edge_index:np.ndarray):
    x_torch = construct_node_feature(x_data, 3)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))
    graph = OrderedData(x=x_torch, edge_index=edge_index, 
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index)
    graph.use_edge_attr = False
    graph.gate = torch.tensor(x_data[:, 1:2], dtype=torch.float)
    return graph

def save_block(args, graphset, batch_size=None, n_hop=None):
    res_list = []
    for graph in graphset:
        design = graph['design']
        if design != 'square_orig':
            continue
        print(design)
        graph_data:dgl.DGLGraph = graph['graph']
        labels = graph['labels']
        cell_num = graph_data.number_of_nodes(ntype='cell')
        hf_cell = torch.zeros((cell_num, 128),requires_grad=True)
        train_nid:torch.Tensor = graph_data.nodes(ntype='cell')
        train_nid = np.array(train_nid.tolist(), dtype=np.int32)
        if batch_size is None:
            batch_size = args.batch_size
        if n_hop is None:
            n_hop = args.n_layer
        sampler = NeighborSampler(graph_data, args.n_layer)
        dataloader = DataLoader(train_nid, args.batch_size, collate_fn=sampler.sample_blocks,shuffle=True,num_workers=args.num_worker)

        for blocks in dataloader:
            src_aig_node = blocks[0].srcdata[dgl.NID]['aig']
            if len(src_aig_node) == 0:
                continue
            subg = dgl.node_subgraph(graph['graph'], {'aig': src_aig_node,'cell':torch.zeros(0,dtype=torch.int64)})
            cell_seeds = blocks[-1].srcdata[dgl.NID]['cell']
            gate_type = subg.ndata['gate_type']['aig']
            if subg is None:
                continue
            try:
                subg = dgl.to_homogeneous(subg)
            except TypeError:
                print(src_aig_node)
                print(subg)
            x_data = []
            for node in subg.nodes():
                gate = gate_type[node]
                x_data.append([int(node.data), int(gate.data)])
            edge_index = subg.edges()
            dg_graph = parse_aig2dg(np.array(x_data), np.array(list(torch.stack(edge_index,dim=0).transpose(0,1))))
            eps = graph['eps']
            hf:torch.Tensor = graph['hf'][src_aig_node+eps['aig']]
            hs = graph['hs'][src_aig_node+eps['aig']]
            block_dict = {'blocks':blocks, 'hf':hf, 'hs':hs, 'dg_graph':dg_graph ,'design':design, 'lable': labels[cell_seeds,:]}
            res_list.append(block_dict)
    torch.save(res_list, f'/home/zhoulingfeng/data/cutLibData/save_subgraph/blockset_hop{n_hop}_batch{batch_size}.pt')

if __name__ == '__main__':
    opt = get_opt()
    dataset_dir = os.path.join(opt.dataset_dir,'graph_set_with_dg.pt')
    graphset = torch.load(dataset_dir)
    save_block(opt, graphset,16,4)