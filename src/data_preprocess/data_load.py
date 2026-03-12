from collections import deque
import copy
import math
import subprocess
from typing import List, Tuple, Dict
# import aiger
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

n_inputs=-1
n_and=-1
n_outputs=-1
n_map = -1
n_not = -1
n_thread = 10
n_not_lock = threading.Lock()

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

def get_aig_gate_type_gpu(nodeId):
    # -1 Const | 0 PI | 1 ANd | 2 PO
    global n_inputs,n_and,n_outputs
    nodeId = int(nodeId)
    if nodeId == 0:
        return -1
    elif nodeId <= n_inputs:
        return 0
    else:
        return 1

def isPi(nodeId):
    # -1 Const | 0 PI | 1 ANd | 2 PO
    global n_inputs,n_and,n_outputs
    nodeId = int(nodeId)
    if nodeId < n_inputs:
        return 1
    return 0

def load_header(aig_path):
    global n_inputs,n_and,n_outputs, n_not, n_map
    abc_cmd = f"read {aig_path}; ps"
    res = subprocess.run(['abc', '-c', abc_cmd], capture_output=True, text=True)
    header = re.search(r"i/o\s*=\s*(\d+)\s*/\s*(\d+).*?and\s*=\s*(\d+)", res.stdout)
    n_inputs = int(header.group(1))
    n_outputs = int(header.group(2))
    n_and = int(header.group(3))
    n_map = -1
    n_not = -1
    return n_inputs,n_and,n_outputs

def map_aig2mapper(mapper_path):
    global n_map
    aig2mapper = {}
    map_id = -1
    aig_id = -1
    aig2mapper[0] = -2
    for i in range(n_inputs):
        aig2mapper[i+1] = i
    with open(mapper_path, 'r') as f:
        for line in f.readlines():
            info = line[:-1]
            match = re.search(r'(\d+\.\d+|\d+)', info)
            if info.startswith('Map'):
                map_id = match.group()
            elif info.startswith('AIG'):
                aig_id = match.group()
                if int(aig_id) in aig2mapper.keys():
                    assert 0, "Get Repeat AIG ID"
                aig2mapper[int(aig_id)] = int(map_id)
            else:
                assert 0, f"Get Unrecognizable Line:{info}"
    n_map = int(map_id)+1
    return aig2mapper

def convertId2Mapper(aig_id, aig2mapper, mode):
    global n_inputs, n_and, n_outputs
    if mode == 'cpu':
        gate_type = get_aig_gate_type(aig_id)
    else:
        gate_type = get_aig_gate_type_gpu(aig_id)
    if gate_type == 2:
        return n_and + aig_id
    else:
        return aig2mapper[aig_id]
    
def load_abc_aig(abcAig_path, aig2mapper, mode='cpu')->dict:
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
                
            # excute po
            # elif len(nodeIds)==3:
            #     nodeIds[0] = convertId2Mapper(nodeIds[0], aig2mapper, mode)
            #     nodeIds[2] = convertId2Mapper(nodeIds[2], aig2mapper, mode)
            #     if aigStruct[nodeIds[0]] is None:
            #         aigStruct[nodeIds[0]] = [(nodeIds[2], nodeIds[1])]
            #     else:
            #         # 去重
            #         if ((nodeIds[2], nodeIds[1])) not in aigStruct[nodeIds[0]]:
            #             aigStruct[nodeIds[0]].append((nodeIds[2], nodeIds[1]))
    # #删除字典中多余的key
    cnt = 0 
    for key in aigStruct:
        if aigStruct[key] is None:
            cnt += 1
    print(f'number of nodes not in aigStruct(Assertion == num_po): {cnt}')
    return aigStruct


def _append_pair_fast(aigStruct: List[List[Tuple[int, int]] | None],
                      src: int, root: int, compl: int) -> None:
    """热路径小函数：给 src 的邻接表追加 (root, compl)；尽量少分支。"""
    bucket = aigStruct[src]
    if bucket is None:
        aigStruct[src] = [(root, compl)]
    else:
        bucket.append((root, compl))
        
def load_abc_aig_v3(
    abcAig_path: str,
    aig2mapper,
    n_inputs: int,
    n_and: int,
    dedup_edges: bool = True,   # 需要边去重就开 True（读完后一次性去重）
    use_convert_cache: bool = True,  # convertId2Mapper 缓存
):
    total_nodes = n_inputs + n_and
    mode: str = "cpu"

    # 邻接表：index 直接用节点ID，None 表示该节点没有作为任何 AND 的 fanin
    aigStruct: List[List[Tuple[int, int]] | None] = [None] * total_nodes

    # 局部绑定，加速属性查找
    conv = convertId2Mapper
    if use_convert_cache:
        cache: Dict[int, int] = {}
        def conv_cached(x: int) -> int:
            y = cache.get(x)
            if y is None:
                y = conv(x, aig2mapper, mode)
                cache[x] = y
            return y
        conv_id = conv_cached
    else:
        def conv_id(x: int) -> int:
            return conv(x, aig2mapper, mode)

    append_pair = _append_pair_fast  # 见下方内联小函数

    with open(abcAig_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            # 快速分割：默认按空白分隔，自动去掉换行
            parts = line.split()
            if not parts:
                continue
            # 只处理 AND 行（5 列）
            if len(parts) != 5:
                continue
            a0 = conv_id(int(parts[0]))
            c0 = int(parts[1])  # 0/1
            a1 = conv_id(int(parts[2]))
            c1 = int(parts[3])  # 0/1
            rt = conv_id(int(parts[4]))

            append_pair(aigStruct, a0, rt, c0)
            append_pair(aigStruct, a1, rt, c1)

    num_none = total_nodes - sum(1 for v in aigStruct if v is not None)
    print(f"number of nodes not in aigStruct(Assertion == num_po): {num_none}")

    if dedup_edges:
        for i, lst in enumerate(aigStruct):
            if lst is None or len(lst) < 2:
                continue
            seen = set()
            new_lst = []
            for pair in lst:
                if pair not in seen:
                    seen.add(pair)
                    new_lst.append(pair)
            aigStruct[i] = new_lst

    return aigStruct


def process_nodes(args):
    nodes, graph, shared_new_graph, shared_po_list = args
    cur_po_collects = set()

    for u in nodes:
        # 分离 NOT=1 和 NOT=0 出边
        not_targets = []
        not_zero_edges = []
        po_not = 0
        if graph[u] is not None:
            for v, not_attr in graph[u]:
                if not_attr == 1:
                    if not v >= n_inputs + n_and:
                        not_targets.append(v)
                    else:
                        po_not = 1
                else:
                    if not v >= n_inputs + n_and:
                        not_zero_edges.append(v)
                    else:
                        cur_po_collects.add(u)
        
        gate_type = isPi(u)
        gate_type = 1- gate_type  
        # 初始化节点 u 的出边列表
        shared_new_graph[u] = [not_zero_edges, 0, -1, gate_type]
        # node : [[neighbors], is_NOT, NOT_node, gate_type]
        
        # 如果有 NOT=1 出边，创建 NOT 节点
        if not_targets or po_not:
            with n_not_lock:
                global n_not
                # print(n_not)
                n_not += 1
                not_node = n_inputs + n_and + n_not
                # print(n_inputs + n_and + n_not)
            if po_not:
                cur_po_collects.add(not_node)
            # 添加边 (u, not_node, NOT=0)
            shared_new_graph[u][0].append(not_node)
            # 为 NOT 节点添加出边
            shared_new_graph[not_node] = [[v for v in not_targets], 1, -1, 2]
            shared_new_graph[u][2] = not_node  # 更新 u 的 NOT 节点索引
        with n_not_lock:
            shared_po_list |= cur_po_collects

def transform_dag_and_threaded(graph, num_threads=n_thread):
    new_graph = {}
    po_nodes = set()
    nodes = list(graph.keys())[:n_inputs+n_and]
    nodes_per_thread = math.ceil(len(nodes) / num_threads)
    node_chunks = [nodes[i:i + nodes_per_thread] for i in range(0, len(nodes), nodes_per_thread)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_nodes, (chunk, graph, new_graph, po_nodes))
            for chunk in node_chunks
        ]
        for future in futures:
            future.result()  # 等待线程完成
    
    # 确保所有节点在 new_graph 中
    for u in graph:
        if u not in new_graph:
            print(f"Node {u} not found in new_graph, initializing with empty list.")
            new_graph[u] = [[], 0, -1, 1]
    
    return new_graph, po_nodes

def transform_dag_and_threaded_v2(graph, num_threads=n_thread):
    new_graph = {}
    po_nodes = set()
    nodes = [i for i in range(n_inputs+n_and)]
    nodes_per_thread = math.ceil(len(nodes) / num_threads)
    node_chunks = [nodes[i:i + nodes_per_thread] for i in range(0, len(nodes), nodes_per_thread)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_nodes, (chunk, graph, new_graph, po_nodes))
            for chunk in node_chunks
        ]
        for future in futures:
            future.result()  # 等待线程完成
    
    # 确保所有节点在 new_graph 中
    for u in range(n_inputs+n_and):
        if u not in new_graph:
            print(f"Node {u} not found in new_graph, initializing with empty list.")
            new_graph[u] = [[], 0, -1, 1]
    
    return new_graph, po_nodes

def read_superGate(abcMap_path:str):
    superGateStruct = {}
    pattern = r"ID:(\d+), AigID:(\d+), Area:([\d.]+), TruthTable:([\d\s,]+), Cut Leaves: ((?:\d+\s*)+)"
    with open(abcMap_path, 'r') as f:
        for line in f.readlines():
            info = line[:-1]
            if info.startswith('Inverter'):
                continue
            match = re.search(pattern, info)
            if match:
                tts = [int(x) for x in match.group(4).replace(' ', '').split(',') if x]
                leaves = list(map(int, match.group(5).split()))
                nVars = len(leaves)
                if nVars <= 5:
                    tt = tts[0]
                else:
                    tt = (tts[1] << 32) | (tts[0] & 0xFFFFFFFF)

                leaves = (tt, leaves, nVars)
                result = {
                    "ID": int(match.group(1)),
                    "Cnt": int(match.group(2)),
                    "Area": float(match.group(3)),
                    "TruthTable": tt,
                    "Cut Leaves": leaves
                }
            else:
                continue
            
            if result['Area'] == 0:
                continue
            sgNode = {'pID':result['ID'], 'pCuts':result['Cut Leaves'], 'Area':result['Area']}
            superGateStruct[result['ID']] = sgNode
    return superGateStruct


def readCut(abcMap_path:str, aig_idmapper):
    superGateStruct = {}
    leaf_pat = r"Truth\s*:\s*(\d+)\s+Leaves\s*:\s*([\d\s]+)"
    with open(abcMap_path, 'r') as f:
        for line in f.readlines():
            line = line[:-1]
            line = line.strip()
            if line.startswith("Node"):
                current_node = int(line.split()[1])
                current_node = aig_idmapper[current_node]
                superGateStruct[current_node] = []
            elif line and current_node is not None:
                matchs = re.search(leaf_pat, line)
                tt = int(matchs.group(1))
                leaves = matchs.group(2).strip().split()
                numbers = [aig_idmapper[int(a)] for a in leaves]
                nVars = len(numbers)
                # if len(numbers) < 4:
                #     continue
                superGateStruct[current_node].append((tt, numbers, nVars))
            else:
                assert 0, f"Get a wrong line {line}"
    return superGateStruct

def build_reverse_graph(aigStruct):
    reverse_graph = {node: ([], copy.deepcopy(attr[1]), 0) for node, attr in aigStruct.items()}
    for node, attrs in aigStruct.items():
        edges = [copy.deepcopy(i) for i in attrs[0]]
        for neighbor in edges:
            reverse_graph[neighbor][0].append(node)
    return reverse_graph

def process_subgraph_nodes_parallel(reverse_graph, root, cut):
    def dfs(node, local_visited, local_subgraph):
        local_visited.append(node)
        local_subgraph.append(node)  # 当前线程找到的节点加入子图集合

        # 如果节点在cut集合中，停止遍历
        if node in cut or isPi(node) == 1:
            return
        
        # 遍历反向图的邻接节点
        for neighbor in reverse_graph[node][0]:
            if neighbor not in local_visited:
                dfs(neighbor, local_visited, local_subgraph)

    local_visited = []  # 线程本地访问集合
    local_subgraph = []  # 线程本地子图集合
    local_visited.append(root)
    
    dfs(root, local_visited, local_subgraph)

    return local_subgraph

def construct_cut_set(aigStruct, aig_idmapper, superPath, cutPath):
    superGateStruct = read_superGate(superPath)
    cutStruct = readCut(cutPath, aig_idmapper)
    reverse_graph = build_reverse_graph(aigStruct)

    cutSet = {}
    have_cell = [0 for a in range(n_inputs+n_and+n_not + 1)]
    for i in cutStruct.keys():
        if len(cutStruct[i]) >= 1:
            have_cell[i] = 1

    def find_list_index(A, B):
        # 将目标列表 A 转换为元组
        A_tuple = tuple(A)
        
        # 使用字典存储 B 中的列表及其索引
        B_dict = {tuple(b[1]): idx for idx, b in enumerate(B)}
        
        # 查找 A_tuple 是否在 B_dict 中
        return B_dict.get(A_tuple, -1)
    
    roots_pal = []
    cuts_pal = []
    for aig_id, cuts in cutStruct.items():
        supergate = superGateStruct.get(aig_id, None)

        if aigStruct[aig_id][2] != -1:
            cell_fanout = len(aigStruct[aig_id][0]) - 1
            cell_fanout += len(aigStruct[aigStruct[aig_id][2]][0])
        else:
            cell_fanout = len(aigStruct[aig_id][0])

        # find supergate in cuts
        if supergate is not None:
            index = find_list_index(supergate['pCuts'][1], cuts)
            if index != -1:
                cutSet[aig_id] = {"cuts": cuts, "supergateIndex": index, "Area": supergate['Area'], "nodes": [], "faninSupers": [], "cellFanout": cell_fanout}
            else:
                # print(f"supergate cut: {supergate['pCuts']} not found in cuts: {cuts} for aig_id: {aig_id}")
                cuts.append(supergate['pCuts'])
                cutSet[aig_id] = {"cuts": cuts, "supergateIndex": len(cuts)-1, "Area": supergate['Area'], "nodes": [], "faninSupers": [], "cellFanout": cell_fanout}
                # print(f"Warning: Supergate {aig_id} not found in cuts.")
        else:
            cutSet[aig_id] = {"cuts": cuts, "supergateIndex": -1, "Area": 0.0, "nodes": [], "faninSupers": [], "cellFanout": cell_fanout}

        # construct logic cone nodes
        for cut in cuts:
            faninSuper = []
            for cut_leaf in cut[1]:
                if have_cell[cut_leaf]:
                    faninSuper.append(cut_leaf)
                # if aigStruct[cut_leaf][2] != -1:
                #     cut[1].append(aigStruct[cut_leaf][2])
            cutSet[aig_id]["faninSupers"].append(faninSuper)
            # print(cut)
            if len(cut[1]) > 0:
                cutSet[aig_id]["nodes"].append(process_subgraph_nodes_parallel(reverse_graph, aig_id, cut[1]))

    return cutSet


def struct2xdata(mapNotStruct:dict):
    # node : [[neighbors], is_NOT, NOT_node, gate_type]
    x_data = sorted(mapNotStruct.keys())
    edge_index = []

    for root_id,node in mapNotStruct.items():
        # print(root_id, node)
        x_data[root_id] = [root_id, node[3]]
        for neighbor in node[0]:
            edge_index.append((root_id, neighbor))
    return x_data,edge_index