import gzip
import json
import numpy as np
import dgl
import torch
import copy
import os
from collections import defaultdict
from data_preprocess.data_process import parse_aig2dg

from subutils.Parser import DeepGateParser
from subutils.config import get_opt
from subutils.tools import Log


def normal_spans(spans):
    new_span = []
    for k,sp_fp in spans.items():
        if k != 'total_dim':
            new_span.append(sp_fp[1])
    return new_span

def convert_feat2dict(feat_list):
    feat_dict = {}
    for feat in feat_list:
        rt_id = feat['rt_id']
        cut_id = feat['cut_id']
        root_key = str(rt_id) + f"_{cut_id}"
        feat_dict[root_key] = feat['feats']
    return feat_dict


def gen_subgraph_wo_PI(opt):
    num_classes=668
    result = []
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log = Log('./log/subgraph_gen.log')
    min_delay = 100000
    max_delay = 0
    all_delays = []
    # cell_mapper_true_designs = {}
    design_list = ["dft_orig", "vga_lcd_orig", "dynamic_node_orig", "des3_area_orig", 
        "wb_conmax_orig", "idft_orig", "multiplier_orig", "ethernet_orig", "sqrt_orig"]
    training_graphs = []
    dgParser = DeepGateParser('./deepgate/pretrained/model.pth')
    dgParser.to_device(device)
    for design in os.listdir(opt.data_root):
        # if design not in design_list:
        #     continue
        log(f"******* Executing {design}")
        design_path = os.path.join(opt.data_root, design)
        if not os.path.isdir(design_path):
            continue

        with open(f"{design_path}/data_2.json", 'r') as f:
            data_dict = json.load(f)
        feat_list = []
        with gzip.open(f"{design_path}/cuts_embeddings.jsonl.gz", 'rt', encoding='utf-8') as f:
            for line in f:
                feat_list.append(json.loads(line))
        
        feat_dict = convert_feat2dict(feat_list)
        feat_dim = len(feat_list[0]['feats']) + 1

        heter_dict = defaultdict(list)
        raw_label = data_dict['cell_struct']
        x_data = data_dict['x_data']
        edge_index = data_dict['edge_index']
        ground_truths = []
        n_node = len(x_data)
        log("DeepGate Feature Generate Done")

        
        # generate edge2supernode
        labels = list()
        gate_gts = torch.empty(0)
        cell_fanouts = []
        cell_sp = n_node
        cell_mapper = {}
        cell_cuts_cnt = defaultdict(int)
        
        feat_nodes = [[0.0] * feat_dim] * n_node
        feat_index = [[-1] * 2] * n_node

        # cell_mapper_true = {}
        for cell_root,cell_gate in raw_label.items():
            if 'abc_delay' in cell_gate.keys() and cell_gate['abc_delay'] > 20000:
                continue
            if 'abc_delay' in cell_gate.keys() and cell_gate['abc_delay'] > 1000:
                continue
            labels_single = [[-1, -1, -1] for i in range(len(cell_gate['cuts']))]
            if 'abc_delay' in cell_gate.keys():
                labels_single[cell_gate['supergateIndex']] = [cell_gate['Area'], cell_gate.get('delay', -1), cell_gate['abc_delay']]
                # print(labels[cell_gate['supergateIndex']])
                all_delays.append(cell_gate['abc_delay'])
                if cell_gate['abc_delay'] < min_delay:
                    min_delay = cell_gate['abc_delay']
                if cell_gate['abc_delay'] > max_delay:
                    max_delay = cell_gate['abc_delay']
            
            # ground_truth = {"cell_id": cell_root, "delay": cell_gate['delay'], "abc_delay": cell_gate['abc_delay'], "nodes": cell_gate['nodes']}
            # ground_truths.append(ground_truth)
            cell_cuts_cnt[cell_root] = len(cell_gate['cuts'])
            for i in range(len(cell_gate['cuts'])):
                cell_nodelist = cell_gate['nodes'][i]
                if len(cell_nodelist) < 4 and i != cell_gate['supergateIndex']:  # skip if less than 4 nodes
                    continue
                cell_fanouts.append(cell_gate['cellFanout'])
                labels.append(labels_single[i])
                cell_mapper[cell_root+f"_{i}"] = n_node
                
                key = cell_root+f"_{i}"
                if key in feat_dict:
                    feats = feat_dict[key]
                    feats.insert(0, cell_gate['cellFanout'])
                    feat_nodes.append(feats)
                    feat_index.append([int(cell_root), i])
                else:
                    log(f"Warning: Feature not found for cell {key}")
                    feat_nodes.append([0.0] * feat_dim)
                    feat_index.append([-1, -1])

                for cell_node in cell_nodelist:
                    heter_dict[('aig', 'mapper', 'cell')].append([cell_node, n_node])
                # if(n_node-cell_sp == 168):
                # print(n_node-cell_sp, n_node, cell_nodelist)
                n_node += 1
        # training_graphs.append({"design": design, "ground_truths": ground_truths,"x_data": x_data, "edge_index": edge_index})
        # cell_mapper_true_designs[design] = cell_mapper_true
        #construct cell2cell connect
        for cell_root,cell_gate in raw_label.items():
            cell_fanins = cell_gate['faninSupers']
            for ind,fanin_cell_list in enumerate(cell_fanins):
                root_key = str(cell_root) + f"_{ind}"
                if root_key not in cell_mapper:
                    continue
                root_id = cell_mapper[root_key]

                for fanin_cell in fanin_cell_list:
                    fanin_cell_str = str(fanin_cell)
                    if fanin_cell_str not in cell_cuts_cnt:
                        continue
                    for i in range(cell_cuts_cnt[fanin_cell_str]):
                        fanin_key = fanin_cell_str + f"_{i}"
                        if fanin_key not in cell_mapper:
                            continue
                        fanin_id = cell_mapper[fanin_key]
                        heter_dict[('cell', 'cell_edge', 'cell')].append([fanin_id, root_id])
        cell_num = n_node - cell_sp
        heter_dict[('aig', 'and', 'aig')]=edge_index
        # print(heter_dict)
        heter_graph:dgl.DGLGraph = dgl.heterograph(heter_dict)

        heter_graph.nodes['cell'].data['feats'] = torch.tensor(feat_nodes, dtype=torch.float64)
        heter_graph.nodes['cell'].data['index'] = torch.tensor(feat_index, dtype=torch.float64)

        labels = torch.tensor(labels,dtype=torch.float32,requires_grad=False)
        log("Subgraph Generate Done")
        log(f"Labels: {len(labels)}, Cell Num: {cell_num}")

        filtered_graph,eps,hs,hf = dgParser.parse_heter_graph(heter_graph, x_data, edge_index, cell_sp, cell_num)
        result.append({'design':design, 'graph':filtered_graph, 'eps':eps,'labels':labels, 'hs':hs, 'hf':hf, 'cell_fanouts':cell_fanouts, 'gate_gts':gate_gts})

        # del dgParser
        torch.cuda.empty_cache()

        log(f"Current min_delay: {min_delay}, max_delay: {max_delay}")
    log(f"min_delay: {min_delay}, max_delay: {max_delay}")
    mean_delay = np.mean(all_delays)
    std_delays = np.std(all_delays)
    all_delays = np.array(all_delays)
    delay_norm = (all_delays-mean_delay)/std_delays
    norm_min = np.min(delay_norm)
    log(f"mean_delay: {mean_delay}, std_delays: {std_delays}, norm_min: {norm_min}")
    torch.save(result,os.path.join(opt.dataset_dir,f'graph_with_dg_cell2cell_{opt.case}.pt'))
    log(f"Subgraph generation completed. Results saved to {opt.dataset_dir}/graph_with_dg_cell2cell_{opt.case}.pt")
    # torch.save(cell_mapper_true_designs, os.path.join(opt.dataset_dir,'cell_mapper_true_designs.pt'))
    # torch.save(training_graphs, os.path.join(opt.dataset_dir,'gcn_graphset.pt'))

if __name__=='__main__':
    opt = get_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.makedirs(opt.dataset_dir, exist_ok=True)
    gen_subgraph_wo_PI(opt)


