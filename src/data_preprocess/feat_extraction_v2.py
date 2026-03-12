import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

class CompleteCutFeatureExtractor:
    def __init__(self, inverter_weight=1.5):
        """
        raw_data_dict: {node_id: [(fanout_id, edge_type), ...]}
        inverter_weight: Weight for NOT edges (default 1.5)
        """
        self.inv_weight = inverter_weight
        
    def _build_weighted_graph(self, raw_data=None):
        if raw_data:
            self.raw_data = raw_data
        G = nx.DiGraph()
        for node_idx, fanouts in self.raw_data.items():
            if not G.has_node(node_idx):
                G.add_node(node_idx)
            for (fanout_idx, edge_type) in fanouts:
                # type=1 (NOT) -> weight=1.5, type=0 (BUF) -> weight=1.0
                w = self.inv_weight if edge_type == 1 else 1.0
                G.add_edge(node_idx, fanout_idx, type=edge_type, weight=w)
        return G

    # --- 1. 基础与时序 (Basic & Timing) ---
    def get_timing_stats(self):
        if self.num_nodes == 0: return [0, 0, 0]
        try:
            # Weighted Depth: 考虑反相器代价的最长路径 (Delay 代理)
            w_depth = nx.dag_longest_path_length(self.G, weight='weight')
            # Topo Depth: 纯逻辑级数
            t_depth = nx.dag_longest_path_length(self.G, weight=None)
        except: w_depth, t_depth = 0, 0
        # Inv Penalty: 反相器带来的额外延迟
        return [w_depth, t_depth, w_depth - t_depth]

    def get_basic_stats(self):
        # Node Count: 面积代理
        return [self.num_nodes]

    # --- 2. 结构统计 (Structural Stats) ---
    def get_structural_stats(self, topo_depth):
        if self.num_nodes == 0: return [0, 0]
        # Avg Width: 并行度 (节点/深度)
        avg_width = self.num_nodes / (topo_depth + 1.0)
        # Max Fanout: 内部最大负载
        out_degrees = [d for n, d in self.G.out_degree()]
        max_fanout = max(out_degrees) if out_degrees else 0
        return [avg_width, max_fanout]

    def get_inverter_stats(self):
        if self.num_edges == 0: return [0, 0]
        # Inv Ratio: 逻辑极性复杂度
        inv_cnt = sum(1 for u, v, d in self.G.edges(data=True) if d.get('type') == 1)
        inv_ratio = inv_cnt / self.num_edges
        # Root Inv: 输出端是否需要反相 (影响输出Delay)
        roots = [n for n, d in self.G.out_degree() if d == 0]
        root_inv = sum(1 for r in roots for u, v, d in self.G.in_edges(r, data=True) if d.get('type')==1)
        return [inv_ratio, root_inv]

    # --- 3. 连续拓扑特征 (Continuous Topology - C Friendly) ---
    def get_continuous_topology(self):
        if self.num_nodes < 2: return [0.0, 0.0, 0.0]
        
        chain_count = 0
        fork_count = 0
        curvature_sum = 0.0
        
        in_deg = dict(self.G.in_degree())
        out_deg = dict(self.G.out_degree())
        
        # Motifs: Chain & Fork
        for n in self.G.nodes():
            n_in = in_deg.get(n, 0)
            n_out = out_deg.get(n, 0)
            chain_count += n_in * n_out
            if n_out > 1:
                fork_count += n_out * (n_out - 1) / 2
                
        # Curvature: Edge Bottleneck
        for u, v, d in self.G.edges(data=True):
            w = d.get('weight', 1.0)
            deg_sum = (in_deg.get(u,0) + out_deg.get(u,0)) + (in_deg.get(v,0) + out_deg.get(v,0))
            if deg_sum > 0:
                curvature_sum += w / deg_sum

        norm = max(1.0, self.num_nodes**2)
        return [chain_count/norm, fork_count/norm, curvature_sum/max(1, self.num_edges)]

    # --- 4. 谱图特征 (Spectral - Conductance based) ---
    def get_laplacian_eigenvalues(self, k=5):
        if self.num_nodes < 2: return [0.0] * k
        # 复制无向图用于谱分析
        G_spec = self.G.to_undirected(as_view=False)
        # 权重倒数: Resistance (Delay) -> Conductance (Affinity)
        for u, v, d in G_spec.edges(data=True):
            w = d.get('weight', 1.0)
            d['weight'] = 1.0 / w if w > 1e-9 else 1.0
            
        try:
            L = nx.normalized_laplacian_matrix(G_spec, weight='weight').astype(float)
            vals = eigsh(L, k=min(k, self.num_nodes-1), which='SM', return_eigenvectors=False)
            vals = np.sort(vals)
        except: vals = np.zeros(0)
        
        res = np.zeros(k)
        res[:len(vals)] = vals
        return res.tolist()

    # --- 5. 离散指纹 (WL Hash - C Logic) ---
    def _c_mix(self, a, b):
        mask = 0xFFFFFFFF
        b &= mask; a &= mask
        term1 = (b + 0x9e3779b9) & mask
        return ((a ^ term1) + ((a << 6) & mask) + ((a >> 2) & mask)) & mask

    def get_wl_signature(self, iterations=2):
        if self.num_nodes == 0: return 0
        labels = {n: (1 if self.G.in_degree(n)==0 else 2) for n in self.G.nodes()}
        
        for _ in range(iterations):
            next_labels = labels.copy()
            for n in self.G.nodes():
                if self.G.in_degree(n) == 0: continue # Skip leaves
                cur_hash = labels[n]
                # 排序保证确定性
                for pred in sorted(self.G.predecessors(n)):
                    p_label = labels[pred]
                    if self.G[pred][n].get('type') == 1:
                        p_label = self._c_mix(p_label, 0xDEADBEEF) # 反相器扰动
                    cur_hash = self._c_mix(cur_hash, p_label)
                next_labels[n] = cur_hash
            labels = next_labels
            
        roots = [n for n, d in self.G.out_degree() if d == 0]
        return labels[roots[0]] if roots else 0

    def extract_all(self, raw_data_dict):
        """
        返回完整特征向量 (13维)
        顺序对应: 
        [Node_Count, Avg_Width, Max_Fanout, Inv_Ratio, Root_Inv_Count, 
         Weighted_Depth, Topo_Depth, Inv_Penalty, Lap_Eig1...Lap_Eig5]
        """
        self.raw_data = raw_data_dict
        self.G = self._build_weighted_graph()
        
        # 预计算一些基础值，避免重复计算
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        
        f_time = self.get_timing_stats()      # [3]
        f_base = self.get_basic_stats()       # [1]
        f_stru = self.get_structural_stats(f_time[1]) # [2]
        f_inv  = self.get_inverter_stats()    # [2]
        f_cont = self.get_continuous_topology() # [3]
        f_lap  = self.get_laplacian_eigenvalues(k=5) # [5]
        
        # 2. 离散特征
        wl_hash = self.get_wl_signature()     # [1]
        
        # 4. 拼接 (共 18 维连续 + 1 维离散)
        # 顺序: 外部 -> 时序 -> 基础 -> 结构 -> 反相器 -> 连续拓扑 -> 谱 -> Hash
        full_vector = f_time + f_base + f_stru + f_inv + f_cont + f_lap + [wl_hash]
        
        names = [
            "Weighted_Depth", "Topo_Depth", "Inv_Penalty", # Timing
            "Node_Count",                    # Basic
            "Avg_Width", "Max_Internal_Fanout", # Structural
            "Inv_Ratio", "Root_Inv_Count",   # Inverter
            "Chain_Density", "Fork_Density", "Avg_Curvature", # Continuous Topology
            "Lap_Eig1", "Lap_Eig2", "Lap_Eig3", "Lap_Eig4", "Lap_Eig5", # Spectral
            "WL_Hash"                        # Identity
        ]
        
        return full_vector, names