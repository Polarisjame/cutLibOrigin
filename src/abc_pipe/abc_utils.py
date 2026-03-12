import numpy as np
from collections import Counter
import dgl
import copy

import torch

n_inputs=-1
n_and=-1
n_outputs=-1
n_map = -1
n_not = -1

def get_aig_gate_type(nodeId):
    # -1 Const | 0 PI | 1 ANd | 2 PO
    global n_inputs,n_and,n_outputs
    nodeId = int(nodeId)
    if nodeId == 0:
        return -1
    elif nodeId <= n_inputs:
        return 0
    elif nodeId <= n_inputs+n_outputs:
        return 2
    else:
        return 1

def convertId2Mapper(aig_id, aig2mapper):
    global n_inputs, n_and, n_outputs
    gate_type = get_aig_gate_type(aig_id)
    if gate_type == 2:
        return n_and + aig_id
    else:
        return aig2mapper[aig_id]
    
def load_feat_graph(abcAig_path):
    global n_inputs,n_and,n_outputs
    aigStruct = dict().fromkeys(range(0, n_inputs + n_and))
    with open(abcAig_path,'r') as aig_ori_f:
        for line in aig_ori_f.readlines():
            nodeIds = line[:-1].strip().split(' ')
            assert len(nodeIds)==5 or len(nodeIds)==3, 'OriAig File Error'
            nodeIds = [int(a) for a in nodeIds]
            # excute and
            if len(nodeIds)==5:
                nodeIds[0] = convertId2Mapper(nodeIds[0], aig2mapper, mode)
                nodeIds[2] = convertId2Mapper(nodeIds[2], aig2mapper, mode)
                nodeIds[4] = convertId2Mapper(nodeIds[4], aig2mapper, mode)
                if aigStruct[nodeIds[0]] is None:
                    aigStruct[nodeIds[0]] = [(nodeIds[4], nodeIds[1])]
                else:
                    # 去重
                    if ((nodeIds[4], nodeIds[1])) not in aigStruct[nodeIds[0]]:
                        aigStruct[nodeIds[0]].append((nodeIds[4], nodeIds[1]))
                if aigStruct[nodeIds[2]] is None:
                    aigStruct[nodeIds[2]] = [(nodeIds[4], nodeIds[3])]
                else:
                    # 去重
                    if ((nodeIds[4], nodeIds[3])) not in aigStruct[nodeIds[2]]:
                        aigStruct[nodeIds[2]].append((nodeIds[4], nodeIds[3]))

def load_abc_aig(abcAig_path)->dict:
    aigStruct = dict()
    global n_and,n_inputs,n_outputs 
    n_and = 0
    n_outputs = 0
    node_done = [0 for i in range(200000)]
    n_inputs = 2000000
    with open(abcAig_path,'r') as aig_ori_f:
        for line in aig_ori_f.readlines():
            nodeIds = line[:-1].strip().split(' ')
            nodeIds = [int(a) for a in nodeIds]

            # excute and
            if len(nodeIds)==5:
                n_and += 1
                src_fanin0 = nodeIds[0]
                comp_fanin0 = nodeIds[1]
                src_fanin1 = nodeIds[2]
                comp_fanin1 = nodeIds[3]
                root_id = nodeIds[4]
                if not node_done[root_id]:
                    node = {'pID':root_id, 'fanin0':src_fanin0, 'fanin1':src_fanin1, 'fanout':[], 'fanin0Not':comp_fanin0, 'fanin1Not':comp_fanin1, 'dfs_trav_ID':-1, 'gate_type':-1}
                    node_done[root_id] = 1
                else:
                    node = aigStruct[root_id]
                    node['fanin0'] = src_fanin0
                    node['fanin1'] = src_fanin1
                    node['fanin0Not'] = comp_fanin0
                    node['fanin1Not'] = comp_fanin1
                aigStruct[root_id] = node
                if not node_done[src_fanin0]:
                    node = {'pID':src_fanin0, 'fanin0':-1, 'fanin1':-1, 'fanout':[root_id],'dfs_trav_ID':-1, 'gate_type':-1}
                    node_done[src_fanin0] = 1
                else:
                    node = aigStruct[src_fanin0]
                    if root_id not in node['fanout']:
                        node['fanout'].append(root_id)
                aigStruct[src_fanin0] = node
                if not node_done[src_fanin1]:
                    node = {'pID':src_fanin1, 'fanin0':-1, 'fanin1':-1, 'fanout':[root_id],'dfs_trav_ID':-1, 'gate_type':-1}
                    node_done[src_fanin1] = 1
                else:
                    node = aigStruct[src_fanin1]
                    if root_id not in node['fanout']:
                        node['fanout'].append(root_id)
                aigStruct[src_fanin1] = node
            else:
                # excute po
                n_outputs += 1
                po_id = nodeIds[2]
                if po_id < n_inputs:
                    n_inputs = po_id
                src_comp = nodeIds[1]
                src_id = nodeIds[0]
                node = {'pID':po_id, 'fanin0':src_id, 'fanin1':-1, 'fanout':[], 'fanin0Not':src_comp, 'fanin1Not':-1, 'dfs_trav_ID':-1, 'gate_type':2}
                aigStruct[po_id] = node
                if not src_id in aigStruct.keys():
                    node = {'pID':src_id, 'fanin0':-1, 'fanin1':-1, 'fanout':[], 'fanin0Not':-1, 'fanin1Not':-1, 'dfs_trav_ID':-1, 'gate_type':1}
                    aigStruct[src_id] = node
                aigStruct[src_id]['fanout'].append(po_id)
        n_inputs -= 1
    return aigStruct

def cnt_gate_type(aigStruct):
    global n_inputs,n_and,n_outputs
    n_inputs = 0
    n_and = 0
    n_outputs = 0
    for key in aigStruct.keys():
        gate_type = get_aig_gate_type(aigStruct[key])
        aigStruct[key]['gate_type'] = gate_type
        if gate_type == 0:
            n_inputs += 1
        elif gate_type == 1:
            n_and += 1
        elif gate_type == 2:
            n_outputs += 1
        else:
            assert 0, f"Get a wrong node ID:{key}"

def construct_fanin0_not(root_id, rootNode, mapStruct, mapNotStruct, hasnot):
    global not_id
    global n_not
    if not hasnot[rootNode['fanin0']]:
        newNode = {'pID':not_id, 'fanin0':rootNode['fanin0'], 'fanin1':-1, 'fanout':[], 'gate_type':2}
        mapNotStruct[rootNode['fanin0']]['notId'] = not_id
        mapNotStruct[not_id] = newNode
        hasnot[rootNode['fanin0']] = 1
        rootNode['fanin0'] = not_id
        not_id += 1
        n_not += 1
    else:
        notNode_id = mapNotStruct[rootNode['fanin0']]['notId']
        rootNode['fanin0'] = notNode_id
    return rootNode
def construct_fanin1_not(root_id, rootNode, mapStruct, mapNotStruct, hasnot):
    global not_id
    global n_not
    if not hasnot[rootNode['fanin1']]:
        newNode = {'pID':not_id, 'fanin0':rootNode['fanin1'], 'fanin1':-1, 'fanout':[], 'gate_type':2}
        mapNotStruct[rootNode['fanin1']]['notId'] = not_id
        mapNotStruct[not_id] = newNode
        hasnot[rootNode['fanin1']] = 1
        rootNode['fanin1'] = not_id
        not_id += 1
        n_not += 1
    else:
        notNode_id = mapNotStruct[rootNode['fanin1']]['notId']
        rootNode['fanin1'] = notNode_id
    return rootNode

def construct_NOT_node_dfs(root_id, visited, aigStruct, mapNotStruct, hasnot):
    global not_id
    if visited[root_id] or aigStruct[root_id]['gate_type'] == -1:
        return
    if aigStruct[root_id]['gate_type'] == 0:
        # exec PIs
        newNode = {'pID':root_id, 'fanin0':-1, 'fanin1':-1, 'fanout':[], 'gate_type':0}
        mapNotStruct[root_id] = newNode
        visited[root_id] = 1
        return
    
    fanin0 = aigStruct[root_id]['fanin0']
    fanin1  = aigStruct[root_id]['fanin1']

    construct_NOT_node_dfs(fanin0,visited, aigStruct, mapNotStruct, hasnot)
    construct_NOT_node_dfs(fanin1,visited, aigStruct, mapNotStruct, hasnot)

    rootNode = copy.deepcopy(aigStruct[root_id])
    visited[root_id] = 1
    if rootNode['fanin0Not'] == 0 and rootNode['fanin1Not'] == 0:
        newNode = {'pID':root_id, 'fanin0':rootNode['fanin0'], 'fanin1':rootNode['fanin1'], 'fanout':[], 'gate_type':rootNode['gate_type']}
        mapNotStruct[root_id] = newNode
    else:
        newNode = {'pID':root_id, 'fanin0':rootNode['fanin0'], 'fanin1':rootNode['fanin1'], 'fanout':[], 'gate_type':rootNode['gate_type']}
        if rootNode['fanin0Not']:
            newNode = construct_fanin0_not(root_id, newNode, aigStruct, mapNotStruct, hasnot)
        if rootNode['fanin1Not']:
            newNode = construct_fanin1_not(root_id, newNode, aigStruct, mapNotStruct, hasnot)
        mapNotStruct[root_id] = newNode
    return 

def convertAigStructure(aigStruct):
    count_pi = 0
    count_and = 0
    cound_po = 0
    aigStructNew = {}
    aig_idmapper = {}
    for key in sorted(aigStruct.keys()):
        if aigStruct[key]['gate_type'] == 0:
            aigStructNew[count_pi] = copy.deepcopy(aigStruct[key])
            aig_idmapper[key] = count_pi
            count_pi += 1
        elif aigStruct[key]['gate_type'] == 1:
            aigStructNew[n_inputs+count_and] = copy.deepcopy(aigStruct[key])
            aig_idmapper[key] = n_inputs + count_and
            count_and += 1
        elif aigStruct[key]['gate_type'] == 2:
            aigStructNew[n_inputs+n_and+cound_po] = copy.deepcopy(aigStruct[key])
            aig_idmapper[key] = n_inputs + n_and + cound_po
            cound_po += 1
        else:
            assert 0, f"Get a wrong node ID:{key}"
    for key in sorted(aigStruct.keys()):
        newId = aig_idmapper[key]
        if not aigStruct[key]['fanin0'] == -1:
            aigStructNew[newId]['fanin0'] = aig_idmapper[aigStruct[key]['fanin0']]
        if not aigStruct[key]['fanin1'] == -1:
            aigStructNew[newId]['fanin1'] = aig_idmapper[aigStruct[key]['fanin1']]
        # aigStructNew[newId]['fanout'] = [aig_idmapper[a] for a in aigStruct[key]['fanout']]
    return aigStructNew, aig_idmapper


def del_po_node(aigStruct, po_id):
    faninId = aigStruct[po_id]['fanin0']
    aigStruct[faninId]['fanout'].remove(po_id)
    del aigStruct[po_id]
    return aigStruct

def construct_NOT_node(aigStruct):
    global n_and,n_inputs,n_outputs
    global not_id
    global n_not
    n_not = 0
    not_id = n_inputs + n_and
    mapper_po_ids = []
    visited = {a:0 for a in range(n_inputs+n_and)}
    hasnot = [0 for i in range(n_inputs + n_and)]
    mapNotStruct = {}

    for po_id in range(n_inputs+n_and, n_inputs+n_and+n_outputs):
        assert aigStruct[po_id]['gate_type'] == 2, f"Get a wrong PO ID:{po_id}"
        po_node = aigStruct[po_id]
        po_fanin_id = int(po_node['fanin0'])
        if po_fanin_id == -2:
            continue
        construct_NOT_node_dfs(po_fanin_id, visited, aigStruct, mapNotStruct, hasnot)
        if po_node['fanin0Not'] == 0:
            mapper_po_ids.append(po_fanin_id)
        else:
            if hasnot[po_fanin_id]:
                not_node_id = mapNotStruct[po_fanin_id]['notId']
                if not not_node_id in mapper_po_ids:
                    mapper_po_ids.append(not_node_id)
            else:
                newNode = {'pID':not_id, 'fanin0':po_fanin_id, 'fanin1':-1, 'fanout':[], 'gate_type':2}
                mapNotStruct[not_id] = newNode
                mapNotStruct[po_fanin_id]['notId'] = not_id
                mapNotStruct[po_fanin_id]['fanout'] = [not_id]
                hasnot[po_fanin_id] = 1
                mapper_po_ids.append(not_id)
                not_id += 1
                n_not += 1
    # 去重mapper_po_ids
    mapper_po_ids = list(set(mapper_po_ids))
    
    return mapNotStruct, mapper_po_ids, n_not

def dfs_fix_fanout(root_id, parent, mapNotStruct, visited):
    # print(root_id)
    if root_id == -1:
        return
    root_node = mapNotStruct[root_id]
    if parent not in root_node['fanout']:
        mapNotStruct[root_id]['fanout'].append(parent)
    if visited[root_id] or root_node['gate_type'] == 0:
        return
    visited[root_id] = 1

    fanin0 = mapNotStruct[root_id]['fanin0']
    fanin1  = mapNotStruct[root_id]['fanin1']

    dfs_fix_fanout(fanin0, root_id, mapNotStruct,visited)
    dfs_fix_fanout(fanin1, root_id, mapNotStruct,visited)


def check_and_fixFanout(mapNotStruct, mapper_po_ids):
    for id in mapNotStruct.keys():
        mapNotStruct[id]['fanout'] = []
    visited = {a:0 for a in range(n_inputs+n_and+n_not)}
    for po_id in mapper_po_ids:
        po_node = mapNotStruct[po_id]
        visited[po_id] = 1
        dfs_fix_fanout(po_node['fanin0'], po_id, mapNotStruct, visited)
        if po_node['fanin1'] != -1:
            dfs_fix_fanout(po_node['fanin1'], po_id, mapNotStruct, visited)

def struct2xdata(mapNotStruct:dict):
    x_data = sorted(mapNotStruct.keys())
    edge_index = []

    for root_id,node in mapNotStruct.items():
        x_data[root_id] = [root_id, node['gate_type']]
        if node['fanin0'] != -1:
            edge_index.append([node['fanin0'],root_id])
        if node['fanin1'] != -1:
            edge_index.append([node['fanin1'],root_id])
    return x_data,edge_index

def dfs_cell(root_id, mapNotStruct, visited, cell_node:dict, cuts):
    if root_id==-1 or root_id in cuts or mapNotStruct[root_id]['gate_type'] == 0 or root_id in cell_node['nodes']:
        return

    rootNode = mapNotStruct[root_id]
    visited[root_id] = 1
    fanin0 = rootNode['fanin0']
    fanin1 = rootNode['fanin1']

    dfs_cell(fanin0,mapNotStruct,visited,cell_node,cuts)
    dfs_cell(fanin1,mapNotStruct,visited,cell_node,cuts)

    # assert root_id not in cell_node['nodes'], f"get repeat cell node {root_id}"
    cell_node['nodes'].append(root_id)


def construct_cell_set(mapNotStruct, superGateStruct, full_node):
    cellStruct={}
    visited = {a:0 for a in range(n_inputs+n_and+n_not)}
    have_cell = [0 for a in range(n_inputs+n_and+n_not)]
    for i in superGateStruct.keys():
        if len(superGateStruct[i]) >= 1:
            have_cell[i] = 1
    for cell_root, cut_sets in superGateStruct.items():
        # cell_root = aig_idmapper[cell_root_old]
        if visited[cell_root] and cell_root != full_node:
            continue
        cut_len = len(cut_sets)
        # sample cuts
        if cut_len > 1 and cell_root != full_node:
            cut_sets_ids = np.random.choice([i for i in range(cut_len)], size=1, replace=False)
            cut_sets = [cut_sets[i] for i in cut_sets_ids]
        for cut_single in cut_sets:
            cell_node = {'pID':cell_root, 'nodes':[], 'leaves':copy.deepcopy(cut_single), 'cellFanout':None, 'faninSuper':[]}
            cell_cuts = cut_single
            for cut in cell_cuts:
                if have_cell[cut]:
                    cell_node['faninSuper'].append(cut)
                if 'notId' in mapNotStruct[cut]:
                    cell_cuts.append(mapNotStruct[cut]['notId'])
            root_node = mapNotStruct[cell_root]
            cell_node['cellFanout'] = len(root_node['fanout'])
            cell_node['nodes'].append(cell_root)
            dfs_cell(root_node['fanin0'],mapNotStruct,visited,cell_node,cell_cuts)
            dfs_cell(root_node['fanin1'],mapNotStruct,visited,cell_node,cell_cuts)
            if cell_root not in cellStruct.keys():
                cellStruct[cell_root] = [cell_node]
            else:
                cellStruct[cell_root].append(cell_node)
        visited[cell_root] = 1
    return cellStruct

def construct_cell_single(mapNotStruct, cellStruct, have_cell, cell_root, cell_leaves):
    cell_node = {'pID':cell_root, 'nodes':[], 'leaves':copy.deepcopy(cell_leaves), 'cellFanout':None, 'faninSuper':[]}
    visited = {a:0 for a in range(n_inputs+n_and+n_not)}
    cell_cuts = cell_leaves
    for cut in cell_cuts:
        if have_cell[cut]:
            cell_node['faninSuper'].append(cut)
        if 'notId' in mapNotStruct[cut]:
            cell_cuts.append(mapNotStruct[cut]['notId'])
    root_node = mapNotStruct[cell_root]
    cell_node['cellFanout'] = len(root_node['fanout'])
    cell_node['nodes'].append(cell_root)
    dfs_cell(root_node['fanin0'],mapNotStruct,visited,cell_node,cell_cuts)
    dfs_cell(root_node['fanin1'],mapNotStruct,visited,cell_node,cell_cuts)
    if cell_root not in cellStruct.keys():
        cellStruct[cell_root] = [cell_node]
    else:
        cellStruct[cell_root].append(cell_node)
    return cellStruct

def get_cut_ind(cell_mapper, cut_root, cut_leaves, cellStruct):
    if not cut_root in cellStruct.keys():
        return -1, False
    cell_cuts = cellStruct[cut_root]
    find = False
    cut_leaves_counter = Counter(cut_leaves)
    for ind,cut in enumerate(cell_cuts):
        if Counter(cut['leaves']) == cut_leaves_counter or Counter(cut['leaves']) <= cut_leaves_counter:
            find = True
            break
    cut_ind = cell_mapper[cut_root*20+ind]
    return cut_ind, find
            

def read_superGate(abcMap_path:str, aig_idmapper):
    superGateStruct = {}
    with open(abcMap_path, 'r') as f:
        for line in f.readlines():
            line = line[:-1]
            line = line.strip()
            if line.startswith("Node"):
                current_node = int(line.split()[1])
                current_node = aig_idmapper[current_node]
                superGateStruct[current_node] = []
            elif line and current_node is not None:
                numbers = [aig_idmapper[int(a)] for a in line.split()]
                superGateStruct[current_node].append(numbers)
            else:
                assert 0, f"Get a wrong line {line}"
    return superGateStruct


class NeighborSampler(object):
    def __init__(self, num_hops, device=None):
        """
        g 为 DGLGraph；
        fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
        """
        self.num_hops = num_hops
        self.device = device


    def sample_blocks(self, g, seeds:torch.Tensor):
        blocks = []
        seeds = seeds.clone().detach().to(torch.int64)
        seeds =  {'cell':seeds}
        for cnt in range(self.num_hops): 
            frontier = dgl.in_subgraph(g,seeds)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]['aig']
            seeds = {'aig':seeds}
            blocks.insert(0, block)
        return blocks