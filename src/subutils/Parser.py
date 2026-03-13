import gzip
import os
import subprocess
import json
import sys
from typing import Iterable
import deepgate
from cffi import FFI
import shutil
import dgl
import time
import copy
import torch

from data_preprocess.feat_extraction_v3 import CompleteCutFeatureExtractor
# from data_preprocess.feat_extraction_v2 import CompleteCutFeatureExtractor
sys.setrecursionlimit(3000)
from data_preprocess.data_load import *
# from src.data_process.data_load import *
from subutils.tools import Log
from subutils.utils import canonize
from data_preprocess.data_process import parse_aig2dg

def find_tt_lib_files(liberty_path):
    tt_lib_files = []
    for root, _, files in os.walk(liberty_path):  # 遍历目录
        for file in files:
            if "TT" in file and file.endswith(".lib"):  # 筛选包含 'TT' 且以 .lib 结尾的文件
                tt_lib_files.append(os.path.join(root, file))  # 添加完整路径
    return tt_lib_files

def shutil_move_wo_except(abc_out_file,output_path):
    if not os.path.exists(abc_out_file):
        return False
    file_name = abc_out_file.split('/')[-1]
    if os.path.exists(os.path.join(output_path, file_name)):
        os.remove(os.path.join(output_path, file_name))
    shutil.move(abc_out_file, output_path)
    return True

def build_featExtraction_graph(transformed_graph):
    new_graph = {}
    for node in transformed_graph.keys():
        transformed_list = transformed_graph[node]
        if transformed_list[1]:
            continue
        new_graph[node] = []
        for fanout in transformed_list[0]:
            if transformed_graph[fanout][1]:
                continue
            new_graph[node].append((fanout,0))
        if transformed_list[2] > -1:
            for fanout in transformed_graph[transformed_list[2]][0]:
                assert transformed_graph[fanout][1] == 0
                new_graph[node].append((fanout, 1))
    return new_graph

def construct_subg_of_cut(nodes, feat_graph, transformed_graph):
    subg = {}
    for node in nodes:
        if transformed_graph[node][1]:
            continue
        fanout_list = feat_graph[node]
        subg[node] = []
        for fanout_comp in fanout_list:
            if fanout_comp[0] in nodes:
                subg[node].append(fanout_comp)
    return subg

def morton_total_bits_from_spans(bits_cfg: Dict[str,int], spans: Dict[str,int]) -> int:
    total = 0
    for name, b in bits_cfg.items():
        s, e = spans[name]
        total += (e - s) * int(b)
    return total

def dump_cuts_jsonl_gz(path: str, records: Iterable[dict]):
    """records: 迭代器，每个元素就是上面的那条 dict"""
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write('\n')

class AigParser:
    def __init__(self, out_dir, log_dir) -> None:
        self.proj_dir = './'
        self.abc_path = './abc/abc'
        self.log = Log(log_dir)
        self.out_dir = out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    def _run_abc(self, abc_script, timeout=1800):
        self.log(f"{self.abc_path} -q '{abc_script}'")
        try:
            completed = subprocess.run(
                [self.abc_path, '-L', self.proj_dir, '-q', abc_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self.log("Process Expired Time Limit")
            return False

        self.log(f"abc exited with code {completed.returncode}")
        return completed.returncode == 0

    def _move_logs(self, output_path, filenames):
        for filename in filenames:
            src = os.path.join(self.proj_dir, filename)
            if not shutil_move_wo_except(src, output_path):
                self.log(f"Skip missing log file: {src}")

    def process(self,aig_file_path, liberty_path = '/home/share/platform/asap7/lib'):
        # lib_file = find_tt_lib_files(liberty_path)
        lib_file = ['./lib/asap7.lib']
        aig_name = aig_file_path.split('/')[-1].split('.')[0]
        output_path = os.path.join(self.out_dir,aig_name)
        os.makedirs(output_path, exist_ok=True)

        abc_prefix = ''.join(f'read {lib}; ' for lib in lib_file)

        self._run_abc(
            f"{abc_prefix}read {aig_file_path}; rewrite -c; map; write_verilog {output_path}/map_verilog.v"
        )
        self._move_logs(
            output_path,
            [
                'abc_aig.log',
                'cut_trave.log',
                'abc_map.log',
                'aig2map.log',
                'abc_logic2netlist.log',
                'abc_superCreate.log',
                'abc_write_verilog.log',
            ],
        )

        self._run_abc(f"{abc_prefix}read {aig_file_path}; map; topo; stime")
        self._move_logs(output_path, ['abc_logictopo.log', 'abc_stime_report.log'])

    # def deepGate_processor(self, x_data, edge_index, circuit_name):

    #     x_data = np.array(x_data)
    #     edge_index = np.array(edge_index)
    #     graph = parse_aig2dg(x_data, edge_index)
    #     graph.name = circuit_name
    #     graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
    #     graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
    #     graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
    #     return graph  


class BenchProjParser:
    def __init__(self,dataset_name, dataset_root, bench_path):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.dataset_aig_path = os.path.join(dataset_root,'aig_strashed')
        self.abc_path = '../abc/abc'
        if not os.path.exists(self.dataset_aig_path):
            os.mkdir(self.dataset_aig_path)
        self.bench_root = os.path.join(dataset_root,bench_path)
        self.abc_path = '../abc/abc'
        self.log = Log('../log/bench_trans_exec.log')

    def process(self):
        for bench_file in os.listdir(self.bench_root):
            log = self.log
            bench_file_path = os.path.join(self.bench_root, bench_file)
            design_name = bench_file.split('.')[0]
            out_aig_name = f"{design_name}.aig"
            out_aig_path = os.path.join(self.dataset_aig_path,out_aig_name)
            abc_cmd = f"abc -c 'read {bench_file_path}; strash; cleanup; write {out_aig_path}'"
            process = subprocess.Popen(abc_cmd, shell=True, stdout=None)
            try:
                process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                log("Process Expired Time Limit")

class Aig2MapParser:
    def __init__(self,dataset_name, log_dir):
        self.map_out_dir = f'/home/zhoulingfeng/data/cutLibData/{dataset_name}'
        self.x_data = None
        self.edge_index = None
        self.log = Log(log_dir)
        
        # ffi = FFI()
        # ffi.cdef("unsigned Abc_TtCanonicize(uint64_t *pTruth, int nVars, char *pCanonPerm);")
        # ffi.cdef("void Abc_TtNormalizeSmallTruth(uint64_t * pTruth, int nVars);")
        # self.ffi = ffi
        # self.lib = ffi.dlopen("./abc_so/libabc.so")

    def process(self, aig_path, data_save_dir):
        log = self.log
        design_name = aig_path.split('/')[-1].split('.')[0]
        log(f"--------executing:{design_name}-----------")
        map_design_out_dir = os.path.join(self.map_out_dir, design_name)
        design_save_dir = os.path.join(data_save_dir,design_name)
        
        if not os.path.exists(design_save_dir):
            os.makedirs(design_save_dir, exist_ok=True)
        
        # if os.path.exists(design_save_dir):
        #     log("Design Has Been Processed")
        #     return
        abcAig_path = f'{map_design_out_dir}/abc_aig.log'
        aig2map_path = f'{map_design_out_dir}/aig2map.log' 
        abcMap_path = f'{map_design_out_dir}/abc_map.log'
        abcCut_path = f'{map_design_out_dir}/cut_trave.log'

        # V2
        n_inputs,n_and,n_outputs = load_header(aig_path)
        aig2mapper = map_aig2mapper(aig2map_path)
        aigStruct = load_abc_aig(abcAig_path, aig2mapper)
        transformed_graph, po_nodes = transform_dag_and_threaded(aigStruct)
        po_nodes = list(po_nodes)
        cellStruct = construct_cut_set(transformed_graph, aig2mapper, abcMap_path, abcCut_path)
        x_data,edge_index = struct2xdata(transformed_graph)

        # pTruth = self.ffi.new("uint64_t[]", 1)   # 只分配一次
        records = []

        feat_graph = build_featExtraction_graph(transformed_graph)
        extractor = CompleteCutFeatureExtractor(feat_graph, inverter_weight=1.5)

        do_cnt = 1
        for cut_rt in cellStruct.keys():
            cuts = cellStruct[cut_rt]['cuts']
            nodes = cellStruct[cut_rt]['nodes']
            # print(cuts, nodes, sep='\n', end='\n------------------\n')
            start = time.perf_counter()
            supergate_index = cellStruct[cut_rt]['supergateIndex']

            if supergate_index == -1:
                continue

            # extract features for selected cut
            # tt, cut, nVars = cuts[supergate_index]
            node = nodes[supergate_index]
            subg = construct_subg_of_cut(node, feat_graph, transformed_graph)
            features, feat_names = extractor.extract_all(subg)
            rec = {
            "rt_id": cut_rt,
            "cut_id": supergate_index,
            "feats": features,
            "subgraph": subg,
            }
            cellStruct[cut_rt]['feats'] = rec
            stop = time.perf_counter()
            if do_cnt:
                log(f"extract feat elapse for one node: {stop-start}")
                do_cnt=0

        dump_cuts_jsonl_gz(f'{design_save_dir}/cuts_embeddings.jsonl.gz', records)
        self.x_data = x_data
        self.edge_index = edge_index

        data = {
            'x_data': x_data,
            'edge_index': edge_index,
            'pos': po_nodes,
            'cell_struct': cellStruct
        }
        # label = superGateStruct

        if not os.path.exists(design_save_dir):
            os.mkdir(design_save_dir)
        with open(f'{design_save_dir}/data.json', 'w') as f:
            f.truncate()
            json.dump(data, f)
        # with open(f'{design_save_dir}/label.json', 'w') as f:
        #     f.truncate()
        #     json.dump(label, f)

class DelayParser():
    def __init__(self, data_dir, log_dir):
        """

        Args:
            data_dir (str): ABC Log Save Dir
            log_dir (str): Exec Log Save Dir
        """
        self.data_save_dir = data_dir
        self.log_dir = log_dir
        self.logger = Log(log_dir)
        self.newNodeMapperPattern = r"Old Node Id:(\d+), New Node ID:\s*(\d+)"
        self.gate_verilogIdPattern = r"Current Node Id:\s*(\d+).*?Gate Name:\s*([\w\d_]+).*?Gate Verilog Id:\s*(\w+).*?Gate Out:\s*(\w+)"
        self.gate_verilogPinPattern = r"Gate Pin:\s*(\w+)"
        self.super_superPattern = r"Creating Super:\s*(\d+)\s*Node"
        self.super_gatePattern = r"Node Id:(\d+) "        
        self.connectPattern = r"\((.*?)\)"
        self.instancePattern = r"\(INSTANCE\s+(\w+)\)"
        #self.iopathPattern = r"\s*\(IOPATH\s+(\w+)\s+(\w+)\s+\((-?\d+\.\d+)::(-?\d+\.\d+)\)"
        self.iopathExtractor = re.compile(r"\(IOPATH\s+(\w+)\s+(\w+)\s+\(([-\d.]+)::([-\d.]+)\)(?:\s+\(([-\d.]+)::([-\d.]+)\))?\)")

        self.superGate_Gates = {}
        self.gateSets = {}
        self.edges = []
        self.verilog_mapper = {} # gateId -> VerilogId
        self.gate_mapper = {} # gateId -> topoGateId
        self.abc_gate_delay = {}
    
    def reset(self):
        self.superGate_Gates = {}
        self.gateSets = {}
        self.edges = []
        self.verilog_mapper = {} # gateId -> VerilogId
        self.gate_mapper = {} # gateId -> topoGateId

    def process_abc_log(self, design):
        """Get SuperGate Sets

        Args:
            design (str): design_name
        """
        design_dir = os.path.join(self.data_save_dir, design)
        abc_spuerLog = os.path.join(design_dir, 'abc_logic2netlist.log')
        newNodeMapper = {}
        oldNodeMapper = {}
        with open(abc_spuerLog) as f:
            superLines = f.readlines()
        for line in superLines:
            line = line[:-1]
            re_match = re.match(self.newNodeMapperPattern, line)
            old_id = int(re_match.group(1))
            new_id = int(re_match.group(2))
            newNodeMapper[old_id] = new_id
            oldNodeMapper[new_id] = old_id


        abc_writeVerilog_path = os.path.join(design_dir, 'abc_write_verilog.log')
        with open(abc_writeVerilog_path, 'r') as f:
            superLines = f.readlines()

        gate2VerilogMapper = {}
        for line in superLines:
            line = line[:-1]
            matchs = re.search(self.gate_verilogIdPattern, line)
            gate_id = int(matchs.group(1))
            gate_name = matchs.group(2)
            verilog_id = matchs.group(3)
            gateOutPort = matchs.group(4)
            gatePins = re.findall(self.gate_verilogPinPattern, line)
            assert gate_id not in gate2VerilogMapper.keys(), f"Get Repeat Gate ID: {gate_id}"
            gate2VerilogMapper[gate_id] = verilog_id
            self.verilog_mapper[verilog_id] = oldNodeMapper[gate_id]
            newGate = {'gateId':verilog_id, 'gateName':gate_name, 'Pins':gatePins, 'Out':gateOutPort, 'IOPathDelay':{}, 'edges':[], 'input_edges':[]}
            self.gateSets[verilog_id] = newGate

        abc_spuerLog = os.path.join(design_dir, 'abc_superCreate.log')
        with open(abc_spuerLog, 'r') as f:
            superLines = f.readlines()
        for line in superLines:
            line = line[:-1]
            if 'Super' in line and 'delay' not in line:
                superId = int(re.match(self.super_superPattern, line).group(1))
                if superId not in self.superGate_Gates.keys():
                    self.superGate_Gates[superId]=[]
            elif 'delay' in line:
                continue
            else:
                gateId = int(re.search(self.super_gatePattern, line).group(1))
                if gateId not in newNodeMapper.keys():
                    self.logger(f"get dangling nodes {gateId}")
                    continue
                verilog_id = gate2VerilogMapper[newNodeMapper[gateId]]
                self.superGate_Gates[superId].append(verilog_id)
        Gates_temp = {}
        for key in self.superGate_Gates.keys():
            if len(self.superGate_Gates[key]) == 0:
                self.logger(f"Delete Dangling Super {key}")
            else:
                Gates_temp[key] = copy.deepcopy(self.superGate_Gates[key])
        del self.superGate_Gates
        self.superGate_Gates = Gates_temp

    def stime_parser(self, design):
        design_dir = os.path.join(self.data_save_dir, design)
        abc_topolog = os.path.join(design_dir, 'abc_logictopo.log')
        with open(abc_topolog, 'r') as f:
            lines = f.readlines()
        for line in lines:
            text = line[:-1]
            old_id = int(text.split("Old Node Id:")[1].split(",")[0].strip())
            new_id = int(text.split("New Node ID:")[1].strip())
            self.gate_mapper[old_id] = new_id
        
        abc_stimelog = os.path.join(design_dir, 'abc_stime_report.log')
        with open(abc_stimelog, 'r') as f:
            lines = f.readlines()
        curr_gateid = -1
        for line in lines:
            text = line[:-1]
            if "Current Obj" in line:
                if curr_gateid > 0:
                    assert max_delay > 0, "Get UnExpected Delay"
                    self.abc_gate_delay[curr_gateid] = max_delay
                curr_gateid = int(text.split("Current Obj:")[1].strip())
                max_delay = 0
            else:
                delays = text.split(' ')
                for delay in delays:
                    delay = float(delay)
                    if max_delay < delay:
                        max_delay = delay
        if curr_gateid > 0:
            assert max_delay > 0, "Get UnExpected Delay"
            self.abc_gate_delay[curr_gateid] = max_delay

    def process_def(self, design):
        """Get Gates Connection Info

        Args:
            design (str): design
        """
        design_dir = os.path.join(self.data_save_dir, design)
        def_dir = os.path.join(design_dir, 'afterPlace.def')
        with open(def_dir, 'r') as f:
            def_lines = f.readlines()
        nets_lines = []
        nets_detect = False
        connect_item = []
        for line in def_lines:
            if "NETS " in line:
                nets_detect = True
                continue
            if "END NETS" in line:
                nets_detect = False
            if not nets_detect:
                continue
            line = line[:-1].strip()
            if "- " in line:
                if len(connect_item) > 0:
                    nets_lines.append(' '.join(connect_item))
                connect_item = [line]
            else:
                connect_item.append(line)
        if len(connect_item) > 0:
            nets_lines.append(' '.join(connect_item))    
        for connect in nets_lines:
            matchs = re.findall(self.connectPattern, connect)
            srcGatePort = [match for match in matchs if 'Y' in match and 'pi' not in match and 'po' not in match]
            dstGatePort = [match for match in matchs if 'Y' not in match and 'pi' not in match and 'po' not in match]
            assert len(srcGatePort)<=1, f"Get Multi OutPorts: {dstGatePort}"
            if len(dstGatePort) == 0 or len(srcGatePort) == 0:
                continue
            src_gate, src_port = srcGatePort[0].strip().split(' ')
            for dst in dstGatePort:
                dst_gate, dst_port = dst.strip().split(' ')
                self.gateSets[src_gate]['edges'].append([f"{src_gate} {src_port}", f"{dst_gate} {dst_port}"])
                self.gateSets[dst_gate]['input_edges'].append([f"{src_gate} {src_port}", f"{dst_gate} {dst_port}"])
        
        
    
    def process_sdf(self, design):
        design_dir = os.path.join(self.data_save_dir, design)
        sdf_dir = os.path.join(design_dir, 'afterPlace.sdf')
        with open(sdf_dir, 'r') as f:
            sdf_lines = f.readlines()
        interConnectLines = []
        gateDelayLines = []

        strat_cell = False
        
        for line in sdf_lines:
            if "INTERCONNECT" in line and 'pi' not in line and 'po' not in line:
                interConnectLines.append(line[:-1].strip())
                strat_cell=True
                continue
            if not strat_cell:
                continue
            line = line[:-1]
            if "INSTANCE" in line:
                gateId = re.search(self.instancePattern, line).group(1)
            elif "IOPATH" in line:
                if self.iopathExtractor.search(line):
                    # matchs = re.search(iopathPattern, line)
                    matchs = self.iopathExtractor.search(line)
                    edge_1 = matchs.group(1) 
                    edge_2 = matchs.group(2) 
                    assert edge_2=='Y', f"{line} Error"
                    delay_rise = float(matchs.group(4))
                    if matchs.group(6):
                        delay_fall = float(matchs.group(6))
                    else:
                        delay_fall = 0
                    if edge_1 not in self.gateSets[gateId]['IOPathDelay'] or (edge_1 in self.gateSets[gateId]['IOPathDelay'] and max(delay_rise,delay_fall)>self.gateSets[gateId]['IOPathDelay'][edge_1]):
                        self.gateSets[gateId]['IOPathDelay'][edge_1] = max(delay_fall, delay_rise)

        interPattern = re.compile(
            r"\(INTERCONNECT\s+([\w/]+)\s+([\w/]+)\s+\(([-\d.]+)::([-\d.]+)\)(?:\s+\(([-\d.]+)::([-\d.]+)\))?\)"
        )

        for interConnect in interConnectLines:
            matchs = interPattern.search(interConnect)
            gate_1 = matchs.group(1).split('/')
            gate_2 = matchs.group(2).split('/')
            delay_rise = float(matchs.group(4))
            delay_fall =  float(matchs.group(6)) if matchs.group(6) else -1
            if gate_1[1] != 'Y':
                dst = gate_1
                src = gate_2
            else:
                dst = gate_2
                src = gate_1
            src_gate, src_port = src
            dst_gate, dst_port = dst
            edge_name = [' '.join(src), ' '.join(dst)]
            assert edge_name in self.gateSets[src_gate]['edges']
            if 'edge_delay' not in self.gateSets[src_gate].keys():
                edge_delay = [0 for i in range(len(self.gateSets[src_gate]['edges']))]
            else: 
                edge_delay = self.gateSets[src_gate]['edge_delay']
            index = self.gateSets[src_gate]['edges'].index(edge_name)
            edge_delay[index] = max(delay_fall, delay_rise)
            self.gateSets[src_gate]['edge_delay'] = edge_delay

    def superGate_delayProcess(self, superGate_id):
        gates = self.superGate_Gates[superGate_id]
        num_nodes = len(gates)
        gateStructs = [self.gateSets[i] for i in gates]
        edges = []
        external_cells = []
        edge_delays = []
        for gate in gateStructs:
            if len(gate['edges']) == 0 :
                external_cells.append(gate['gateId'])
            else:
                for i in range(len(gate['edges'])):
                    if gate['edges'][i][1].split(' ')[0] in gates:
                        edges.append(gate['edges'][i])
                        edge_delays.append(gate['edge_delay'][i])
                    elif gate['edges'][i][0].split(' ')[0] not in external_cells:
                        external_cells.append(gate['edges'][i][0].split(' ')[0])
        paths = self.find_paths(edges)
        delay_max = 0
        for path in paths:
            path_delay = 0
            prior_gate = None
            for i,gate in enumerate(path):
                if i == 0:
                    gate_delay = max(gateStructs[gates.index(gate)]['IOPathDelay'].values())
                    net_delay = 0
                else:
                    for edge in edges:
                        if prior_gate in edge[0] and gate in edge[1]:
                            src_port = edge[0].split()[1]
                            dst_port = edge[1].split()[1]
                            edge_index = gateStructs[gates.index(prior_gate)]['edges'].index(edge)
                    gate_delay = gateStructs[gates.index(gate)]['IOPathDelay'][dst_port]
                    net_delay = gateStructs[gates.index(prior_gate)]['edge_delay'][edge_index]
                path_delay += (net_delay+gate_delay)
                prior_gate = gate
            if path_delay > delay_max:
                delay_max = path_delay
        for ex_cell in external_cells:
            IO_delay = max(self.gateSets[ex_cell]['IOPathDelay'].values())
            if IO_delay > delay_max:
                delay_max = IO_delay
        return delay_max

    def find_paths(self, edges):
        graph = {}
        for src, dst in edges:
            src_node, src_port = src.split()
            dst_node, dst_port = dst.split()
            if dst_node not in graph:
                graph[dst_node] = []
            graph[dst_node].append(src_node)

        all_nodes = set(graph.keys())
        all_sources = {src.split()[0] for src, _ in edges}
        leaf_nodes = list(all_sources - all_nodes)  
        dst_sources = {dst.split()[0] for _,dst in edges}
        top_nodes = list(dst_sources-all_sources)

        def dfs(node, path, paths):
            if node in leaf_nodes:
                paths.append(path)
                return
            for child in graph[node]:
                dfs(child, [child] + path, paths)

        all_paths = []
        for top in top_nodes:
            dfs(top, [top], all_paths)

        return all_paths

    def stimeDelayProcess(self, superGate_id):
        gates = self.superGate_Gates[superGate_id]
        num_nodes = len(gates)
        gateStructs = [self.gateSets[i] for i in gates]
        edges = []
        external_cells = []
        edge_delays = []
        for gate in gateStructs:
            if len(gate['edges']) == 0 :
                external_cells.append(gate['gateId'])
            else:
                for i in range(len(gate['edges'])):
                    if gate['edges'][i][1].split(' ')[0] in gates:
                        edges.append(gate['edges'][i])
                        edge_delays.append(gate['edge_delay'][i])
                    elif gate['edges'][i][0].split(' ')[0] not in external_cells:
                        external_cells.append(gate['edges'][i][0].split(' ')[0])
        paths = self.find_paths(edges)
        delay_max = 0
        for path in paths:
            path_delay = 0
            for i,gate in enumerate(path):
                old_id = self.verilog_mapper[gate]
                new_id = self.gate_mapper[old_id]
                gate_delay = self.abc_gate_delay[new_id]
                path_delay += gate_delay
            if path_delay > delay_max:
                delay_max = path_delay
        for ex_cell in external_cells:
            old_id = self.verilog_mapper[ex_cell]
            new_id = self.gate_mapper[old_id]
            gate_delay = self.abc_gate_delay[new_id]
            IO_delay = gate_delay
            if IO_delay > delay_max:
                delay_max = IO_delay
        return delay_max

    def collect_cell_names(self, cell_id, gate_name_list):
        verilog_ids = self.superGate_Gates[cell_id]
        gate_ids = []
        for verilog_id in verilog_ids:
            gate = self.gateSets[verilog_id]
            gate_name = gate['gateName']
            gate_id = gate_name_list.index(gate_name)
            gate_ids.append(gate_id)
        return gate_ids

    def process(self, design, data_save_dir):
        self.logger(f"Processing {design}")

        with open(os.path.join(data_save_dir, design, 'data.json'), 'r') as f:
            data_stored = json.load(f)
        cellStruct = data_stored['cell_struct']

        # process log
        self.reset()
        self.process_abc_log(design)
        # self.process_def(design)
        # self.process_sdf(design)
        self.stime_parser(design)
        _,cell_name_list = liberty_parser()

        for cell_id in cellStruct.keys():
            # if 'delay' in cellStruct[cell_id]:
            #     self.logger(f"****** Design {design} has been processed, Skip")
            #     return
            if cellStruct[cell_id]['supergateIndex'] == -1:
                continue
            cell_id = int(cell_id)
            if not cell_id in self.superGate_Gates.keys():
                self.logger(f"{cell_id} Cell Unprocessed, maybe it's a dangling node")
                continue
            # delay = self.superGate_delayProcess(int(cell_id))
            # delay = cellStruct[str(cell_id)]['delay']
            abc_delay = self.stimeDelayProcess(int(cell_id))
            gate_ids = self.collect_cell_names(int(cell_id), cell_name_list)
            if abc_delay == 0:
                self.logger(f"****** Design {design} Cell {cell_id} has Delay 0")
            cell_id = str(cell_id)
            # cellStruct[cell_id]['delay'] = delay
            cellStruct[cell_id]['abc_delay'] = abc_delay
            cellStruct[cell_id]['gate_ids'] = gate_ids
        data_stored['cell_struct'] = cellStruct

        with open(f'{data_save_dir}/{design}/data_2.json', 'w') as f:
            f.truncate()
            json.dump(data_stored, f)


class DeepGateParser():
    def __init__(self,pretrained_path=None, device='cpu'):
        self.model = deepgate.Model()
        self.device = device
        if pretrained_path is not None:
            self.model.load_pretrained(pretrained_path)

    def to_device(self, device):
        self.model.to(device)
        self.device = device

    def extract_aig_feat(self, x_data, edge_index):
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        graph = parse_aig2dg(x_data, edge_index)
        graph = graph.to(self.device)
        hs, hf = self.model(graph) 
        hs=hs.cpu().detach()
        hf=hf.cpu().detach()
        del graph
        torch.cuda.empty_cache()
        return hs,hf
    
    def filter_graph(self, heter_graph:dgl.DGLGraph):
        """
        @param heter_graph: heterogeneous graph
        """
        cell_ind = torch.nonzero(heter_graph.ndata['gate_type']['cell'],as_tuple=True)[0]
        aig_ind = torch.nonzero(heter_graph.ndata['gate_type']['aig'],as_tuple=True)[0]
        indices={'aig':aig_ind, 'cell':cell_ind}
        filtered_graph = dgl.node_subgraph(heter_graph, indices)
        eps = {'aig':copy.deepcopy(aig_ind[0]),'cell':copy.deepcopy(cell_ind[0])}
        return filtered_graph, eps

    def parse_heter_graph(self, heter_graph, x_data, edge_index, cell_sp, cell_num):
            heter_x_gate = {'aig':None, 'cell':[]}
            # construct gate_type feat
            heter_x_gate['cell'] = torch.tensor([0]*cell_sp+[3]*cell_num,dtype=torch.short)
            x_data=np.array(x_data)
            heter_x_gate['aig'] = torch.tensor(x_data[:, 1:2],dtype=torch.short)
            heter_graph.ndata['gate_type'] = heter_x_gate
            filtered_graph, eps = self.filter_graph(heter_graph)
            
            with torch.no_grad():
                hs,hf = self.extract_aig_feat(x_data, edge_index)
            n_input = sum(1 for item in x_data if item[1] == 0)

            # construct topo_level feature
            homo_aig_g = dgl.graph(filtered_graph.edges(etype='and'))
            topo_order = dgl.traversal.topological_nodes_generator(homo_aig_g)
            aig_topo_feat = torch.zeros((filtered_graph.num_nodes(ntype='aig'),1),dtype=torch.int16)
            topo_level = 0
            for topo_list in topo_order:
                aig_topo_feat[topo_list] = torch.ones_like(topo_list,dtype=torch.int16).unsqueeze(1)*topo_level
                topo_level+=1

            cell_topo_feat_list = []
            for root in filtered_graph.nodes(ntype='cell'):
                preds = filtered_graph.predecessors(root,'mapper')
                # print(root, preds, end='  |  ')
                # print(filtered_graph.ndata[dgl.NID]['cell'][int(root)])
                cell_topo_feat = torch.max(aig_topo_feat[preds])+1
                cell_topo_feat_list.append(cell_topo_feat)
            cell_topo_feat = torch.tensor(cell_topo_feat_list,dtype=torch.int16)
            filtered_graph.ndata['topo_feat'] = {'aig':aig_topo_feat, 'cell':cell_topo_feat}
            return filtered_graph, eps, hs, hf
    
def liberty_parser(liberty_path = r'./lib/asap7_libertylog.log'):
    """_summary_ Generate Liberty Embedding
    Returns:
        torch.Tensor: liberty_embedding
    """
    with open(liberty_path, 'r') as f:
        lines = f.readlines()
    line_iter = iter(lines)

    cell_list = []
    cell_name_list = []
    pin_tables = []
    curr_cell = []
    for line in line_iter:
        text = line[:-1]
        if "Curr Cell Name" in text:
            if len(pin_tables) > 0:
                assert len(pin_tables) == 49*4
                curr_cell.append(pin_tables)
                pin_tables = []
            if len(curr_cell) > 0:
                curr_cell = torch.tensor(curr_cell)
                cell_cong = torch.max(curr_cell,dim=0).values
                cell_name_list.append(cell_name)
                cell_list.append(cell_cong)
            curr_cell = []
            cell_name = text.split('Curr Cell Name: ')[1].strip()
        if "Curr Pin " in text:
            if len(pin_tables) > 0:
                assert len(pin_tables) == 49*4
                curr_cell.append(pin_tables)
            pin_tables = []
        if "Curr Phase" in text:
            table_values = next(line_iter,None)[:-1].strip()
            table_values = [float(a) for a in table_values.split(' ')]
            assert len(table_values) == 49
            pin_tables += table_values

    if len(pin_tables) > 0:
        assert len(pin_tables) == 49*4
        curr_cell.append(pin_tables)
        pin_tables = []
    if len(curr_cell) > 0:
        curr_cell = torch.tensor(curr_cell)
        cell_cong = torch.max(curr_cell,dim=0).values
        cell_name_list.append(cell_name)
        cell_list.append(cell_cong)   
        curr_cell = []
    cell_list = torch.stack(cell_list)
    return cell_list, cell_name_list
