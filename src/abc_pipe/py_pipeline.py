from collections import defaultdict
import os
import json
import time
import dgl
import torch
import random
import shutil

from abc_pipe.abc_utils import *
from models.CellPool_wo_lib import CellPool
from subutils.config import argparse
from abc_pipe.sendRec import *
import socket


# 创建TCP服务器
HOST = '127.0.0.1'
PORT = 65436
false_cnt = 0

temp_data_dir = './temp_data_multi'

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def unnormalize_label(delay):
    delay_label = delay
    delay_mean = 152.7085861209273
    delay_std = 176.21307864358835
    delay_norm_min = -0.83461 - 3e-6
    delay_unnorm = (delay_label + delay_norm_min) * delay_std + delay_mean
    return delay_unnorm

def inference_buildData(dgParser, full_node, aig_ori_path, abcMap_path):
    aigStruct = load_abc_aig(aig_ori_path)
    cnt_gate_type(aigStruct)
    aigNew, aig_idmapper = convertAigStructure(aigStruct)
    full_node = aig_idmapper[full_node]
    mapNotStruct, mapper_po_ids, n_not = construct_NOT_node(aigNew)
    check_and_fixFanout(mapNotStruct, mapper_po_ids)
    superGateStruct = read_superGate(abcMap_path, aig_idmapper)

    x_data,edge_index = struct2xdata(mapNotStruct)
    cellStruct = construct_cell_set(mapNotStruct, superGateStruct, full_node)

    # SubGraphGen
    n_node = len(x_data)
    cell_fanouts = []
    cell_sp = len(x_data)
    heter_dict = defaultdict(list)
    cell_mapper = {}

    for cell_root,cell_gate_sets in cellStruct.items():
        for ind,cell_gate in enumerate(cell_gate_sets):
            cell_mapper[cell_root*20+ind] = n_node-cell_sp
            cell_fanouts.append(cell_gate['cellFanout'])
            for cell_node in cell_gate['nodes']:
                heter_dict[('aig', 'mapper', 'cell')].append([cell_node, n_node])
            n_node += 1

    cell_num = n_node - cell_sp
    heter_dict[('aig', 'and', 'aig')]=edge_index
    heter_graph:dgl.DGLGraph = dgl.heterograph(heter_dict)
    filtered_graph,eps = dgParser.parse_heter_graph(heter_graph, x_data, edge_index, cell_sp, cell_num)
    return aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct
    
def inference_model(model, sampler, device, cut_ind, cell_fanouts, filtered_graph):
    # GetLabSamp
    # cut_ind = aig_idmapper[cut_ind]
    blocks = sampler.sample_blocks(filtered_graph, torch.tensor(cut_ind))
    root = blocks[0].srcdata[dgl.NID]['aig']
    fanout = cell_fanouts[blocks[-1].dstdata[dgl.NID]['cell']]
    and_edge_subgraph = dgl.edge_type_subgraph(filtered_graph, [('aig', 'and', 'aig')])
    gate_type = and_edge_subgraph.ndata['gate_type']
    homo_graph = dgl.to_homogeneous(and_edge_subgraph)
    sg = homo_graph  # 子图
    rt_nodes = root
    reachable_nodes_set = root
    for _ in range(int(4)):
        sg = dgl.sampling.sample_neighbors(homo_graph, rt_nodes, fanout=-1) 
        rt_nodes = sg.edges()[0]
        reachable_nodes_set = torch.cat((reachable_nodes_set, rt_nodes))
    reachable_nodes_set = reachable_nodes_set.unique()
    root_index = [torch.where(reachable_nodes_set == x)[0] for x in root]
    root_index = torch.stack(root_index).squeeze()
    subgraph = homo_graph.subgraph(reachable_nodes_set)
    x_data = []
    for cnt, node in enumerate(subgraph.ndata[dgl.NID]):
        gate = gate_type[node]
        x_data.append([cnt, int(gate.data)])
    edge_index = subgraph.edges()
    src, dst = blocks[-1].in_edges(blocks[-1].nodes(ntype='cell'),etype='mapper')
    neighbor_dict = {root.item(): src[dst == root].tolist() for root in blocks[-1].nodes(ntype='cell')}
    dg_graph = parse_aig2dg(np.array(x_data), np.array(list(torch.stack(edge_index,dim=0).transpose(0,1))))

    # Model Inference
    dg_graph = dg_graph.to(device)
    with torch.no_grad():
        hs_c, hf_c = model.deepgate(dg_graph)
        hs_root = hs_c[root_index]
        hf_root = hf_c[root_index]
        hf_root = torch.cat((hf_root,hs_root),dim=-1)
        cell_fanout = torch.tensor(fanout, dtype=torch.int, requires_grad=False).view(-1,1).to(device)
        hf_cell = model.cell_pooling(blocks, hf_root, 256, cell_fanout, neighbor_dict)

        batch_inputs = {'aig':hf_root,'cell':hf_cell}
        pred_res = model(blocks, batch_inputs)
        pred_res = unnormalize_label(pred_res)
    return float(pred_res)


def process(data_json, have_built, dgParser, model, sampler, device):
    # {"stage", "root", "Leaves"}
    global aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct, false_cnt
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError:
        print(f"JSON Decode Error: {data_json}")
        pred_res = -10000
        return have_built, pred_res, False
    stage = data.get("stage", 0)
    Finish = False
    if stage == 'ori':
        root = data.get("root", 0)
        leaves = data.get("Leaves", 0)
        aig_ori_path = os.path.join(temp_data_dir, 'RwrStrucutOri.log')
        abcMap_path = os.path.join(temp_data_dir, 'cut_trave.log')
        if not have_built:
            aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct = inference_buildData(dgParser, root, aig_ori_path, abcMap_path)
            have_built = True
        root_id = aig_idmapper[int(root)]
        leaves_id = [aig_idmapper[int(x)] for x in leaves]
        cut_ind, find = get_cut_ind(cell_mapper, root_id, leaves_id, cellStruct)
        if not find:
            print("No cut found")
            shutil.move(aig_ori_path, os.path.join(temp_data_dir, 'False_Data', f'RwrStrucutOri_{false_cnt}.log'))
            shutil.move(abcMap_path, os.path.join(temp_data_dir, 'False_Data', f'cut_trave_{false_cnt}.log'))
            print(f"收到数据: {data}")
            false_cnt += 1
            pred_res = -10000
            Finish = True
            return have_built, pred_res, Finish
        pred_res = inference_model(model, sampler, device, cut_ind, cell_fanouts, filtered_graph)
        # pred_res = 100
    elif stage == 'update':
        leaves = data.get("Leaves", 0)
        root = data.get("root", 0)
        aig_ori_path = os.path.join(temp_data_dir, 'RwrStrucutUpdated.log')
        abcMap_path = os.path.join(temp_data_dir, 'cut_trave_update.log')
        aig_idmapper_update, cell_fanouts_update, filtered_graph_update, cell_mapper_update, cellStruct_update = inference_buildData(dgParser, root, aig_ori_path, abcMap_path)
        root_id = aig_idmapper_update[int(root)]
        leaves_id = [aig_idmapper_update[int(x)] for x in leaves if x in aig_idmapper_update.keys()]
        cut_ind, find = get_cut_ind(cell_mapper_update, root_id, leaves_id, cellStruct_update)
        if not find:
            print("No cut found")
            shutil.move(aig_ori_path, os.path.join(temp_data_dir, 'False_Data', f'RwrStrucutUpdated_{false_cnt}.log'))
            shutil.move(os.path.join(temp_data_dir, 'RwrStrucutUpdatedBeforeRep.log'), os.path.join(temp_data_dir, 'False_Data', f'RwrStrucutUpdatedBeforeRep_{false_cnt}.log'))
            shutil.move(abcMap_path, os.path.join(temp_data_dir, 'False_Data', f'cut_trave_update_{false_cnt}.log'))
            false_cnt += 1
            print(f"收到数据: {data}")
            pred_res = -10000
            Finish = False
            return have_built, pred_res, Finish
        pred_res = inference_model(model, sampler, device, cut_ind, cell_fanouts_update, filtered_graph_update)
        # pred_res = 100
    elif stage == 'done node':
        aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct = None, None, None, None, None
        have_built = False
        pred_res = None
    elif stage == 'done':
        have_built = False
        Finish = True
        pred_res = None
    else:
        print("Invalid stage")
        pred_res = None
    return have_built, pred_res, Finish

def main(opt):
    global aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct
    dgParser = DeepGateParser('./deepgate/pretrained/model.pth')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dgParser.to_device(device)
    sampler = NeighborSampler(5, 'cuda:0')
    model = CellPool(opt,'prelu')
    ckpt = torch.load(opt.model_ckpt)['checkpoint']
    model.load_state_dict(ckpt,strict=True)
    model.to("cuda:0")
    model.eval()
    have_built = False
    aig_idmapper, cell_fanouts, filtered_graph, cell_mapper, cellStruct = None, None, None, None, None
    # have_built, pred_res, Finish = process(data, have_built)
    # print(have_built, pred_res)
    sock = manage_socket(HOST, PORT)

    while True:
        data = receive_data(sock)
        # print(f"收到数据: {data}")

        # print(have_built)
        have_built, pred_res, Finish = process(data, have_built, dgParser, model, sampler, device)
        # print(have_built, pred_res, Finish)
        if not Finish:
            if pred_res:
                send_data(sock,str(pred_res))
            else:
                send_data(sock, "Receieve")
            # conn.sendall(str(pred_res).encode())
        else:
            send_data(sock,"finish")




if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    opt = argparse.Namespace(**config)
    seed_torch(42)
    main(opt)