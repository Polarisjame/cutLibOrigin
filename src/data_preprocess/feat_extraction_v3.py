import numpy as np
import networkx as nx
from collections import deque

class CompleteCutFeatureExtractor:
    def __init__(self, feat_graph, inverter_weight=1.5, leaf_fanout_mode="total"):
        """
        feat_graph: 全局 AIG 结构，格式 {node: [(fanout, is_not_edge), ...]}
                   边方向必须是 node -> fanout（fanout 表示使用 node 输出作为输入的后继）
        inverter_weight: NOT 边权重
        leaf_fanout_mode:
            - "external": 叶子 fanout 只计 cut 外部（更符合“cut 外部负载”）
            - "total":    叶子 fanout 计全网总 fanout（包含 cut 内部那条）
        """
        self.inv_weight = inverter_weight
        self.leaf_fanout_mode = leaf_fanout_mode

        self.feat_graph = feat_graph
        self._precompute_global_context()

    # -------------------------
    # Global precompute (once)
    # -------------------------
    def _collect_nodes(self, graph):
        nodes = set(graph.keys())
        for u, outs in graph.items():
            for v, _ in outs:
                nodes.add(v)
        return nodes

    def _build_preds_and_degrees(self, graph, nodes):
        preds = {n: [] for n in nodes}
        outdeg = {n: 0 for n in nodes}
        for u, outs in graph.items():
            outdeg[u] = len(outs)
            for v, _ in outs:
                preds[v].append(u)
        # 对于没有出现在 graph.keys() 的节点，outdeg 默认 0 已有
        return preds, outdeg

    def _topo_order_kahn(self, graph, nodes, preds, outdeg):
        indeg = {n: len(preds[n]) for n in nodes}
        q = deque([n for n in nodes if indeg[n] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v, _ in graph.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(topo) != len(nodes):
            raise ValueError("Global graph is not a DAG (cycle detected) or adjacency incomplete.")
        return topo

    def _precompute_global_context(self):
        g = self.feat_graph
        nodes = self._collect_nodes(g)
        preds, outdeg = self._build_preds_and_degrees(g, nodes)
        topo = self._topo_order_kahn(g, nodes, preds, outdeg)

        # PI/PO
        is_pi = {n: (len(preds[n]) == 0) for n in nodes}
        fanout_global = {n: outdeg[n] for n in nodes}  # 全网出度
        POs = [n for n in nodes if outdeg[n] == 0]

        # L_global: longest PI->v level (edge count)
        L = {n: 0 for n in nodes}
        for v in topo:
            if len(preds[v]) == 0:
                L[v] = 0
            else:
                L[v] = max(L[u] + 1 for u in preds[v])

        global_depth = max((L[po] for po in POs), default=max(L.values(), default=0))

        # R_global: longest v->PO reverse level (edge count)
        R = {n: 0 for n in nodes}
        for v in reversed(topo):
            outs = g.get(v, [])
            if len(outs) == 0:
                R[v] = 0
            else:
                R[v] = max(R[w] + 1 for (w, _) in outs)

        # SlackProxy = D - (L+R)
        SlackProxy = {n: (global_depth - (L[n] + R[n])) for n in nodes}

        # # 可选：ReqLevel/SlackProxy2（更像 required-arrival），这里也一并算好（不强制你用）
        # INF = 10**9
        # Req = {n: INF for n in nodes}
        # for po in POs:
        #     Req[po] = global_depth
        # for v in reversed(topo):
        #     outs = g.get(v, [])
        #     if len(outs) == 0:
        #         continue
        #     Req[v] = min(Req[w] - 1 for (w, _) in outs)
        # SlackProxy2 = {}
        # for n in nodes:
        #     if Req[n] >= INF // 2:
        #         SlackProxy2[n] = SlackProxy[n]  # 兜底：不可达 PO 的点（一般不会发生）
        #     else:
        #         SlackProxy2[n] = Req[n] - L[n]

        # 缓存
        self.global_nodes = nodes
        self.global_depth = global_depth
        self.L_global = L
        self.R_global = R
        self.SlackProxy_global = SlackProxy
        # self.SlackProxy2_global = SlackProxy2
        self.fanout_global = fanout_global
        self.is_pi_global = is_pi

    # -------------------------
    # Cut graph build
    # -------------------------
    def _build_weighted_graph(self, raw_data=None):
        if raw_data is not None:
            self.raw_data = raw_data
        G = nx.DiGraph()
        for node_idx, fanouts in self.raw_data.items():
            if not G.has_node(node_idx):
                G.add_node(node_idx)
            for (fanout_idx, edge_type) in fanouts:
                w = self.inv_weight if edge_type == 1 else 1.0
                G.add_edge(node_idx, fanout_idx, type=edge_type, weight=w)
        return G

    # -------------------------
    # Existing feature blocks
    # -------------------------
    def get_timing_stats(self):
        if self.num_nodes == 0:
            return [0, 0, 0]
        try:
            w_depth = nx.dag_longest_path_length(self.G, weight='weight')
            t_depth = nx.dag_longest_path_length(self.G, weight=None)
        except Exception:
            w_depth, t_depth = 0, 0
        return [w_depth, t_depth, w_depth - t_depth]

    def get_basic_stats(self):
        return [self.num_nodes]

    def get_structural_stats(self, topo_depth):
        if self.num_nodes == 0:
            return [0, 0]
        avg_width = self.num_nodes / (topo_depth + 1.0)
        out_degrees = [d for n, d in self.G.out_degree()]
        max_fanout = max(out_degrees) if out_degrees else 0
        return [avg_width, max_fanout]

    def get_inverter_stats(self):
        if self.num_edges == 0:
            return [0, 0]
        inv_cnt = sum(1 for u, v, d in self.G.edges(data=True) if d.get('type') == 1)
        inv_ratio = inv_cnt / self.num_edges

        roots = [n for n, d in self.G.out_degree() if d == 0]
        root_inv = sum(
            1
            for r in roots
            for u, v, d in self.G.in_edges(r, data=True)
            if d.get('type') == 1
        )
        return [inv_ratio, root_inv]

    def get_continuous_topology(self):
        if self.num_nodes < 2:
            return [0.0, 0.0, 0.0]

        chain_count = 0
        fork_count = 0
        curvature_sum = 0.0

        in_deg = dict(self.G.in_degree())
        out_deg = dict(self.G.out_degree())

        for n in self.G.nodes():
            n_in = in_deg.get(n, 0)
            n_out = out_deg.get(n, 0)
            chain_count += n_in * n_out
            if n_out > 1:
                fork_count += n_out * (n_out - 1) / 2

        for u, v, d in self.G.edges(data=True):
            w = d.get('weight', 1.0)
            deg_sum = (in_deg.get(u, 0) + out_deg.get(u, 0)) + (in_deg.get(v, 0) + out_deg.get(v, 0))
            if deg_sum > 0:
                curvature_sum += w / deg_sum

        norm = max(1.0, self.num_nodes ** 2)
        return [chain_count / norm, fork_count / norm, curvature_sum / max(1, self.num_edges)]

    def get_laplacian_eigenvalues(self, k=5):
        if self.num_nodes < 2:
            return [0.0] * k

        G_spec = self.G.to_undirected(as_view=False)
        for u, v, d in G_spec.edges(data=True):
            w = d.get('weight', 1.0)
            d['weight'] = 1.0 / w if w > 1e-9 else 1.0

        try:
            from scipy.sparse.linalg import eigsh
            L = nx.normalized_laplacian_matrix(G_spec, weight='weight').astype(float)
            vals = eigsh(L, k=min(k, self.num_nodes - 1), which='SM', return_eigenvectors=False)
            vals = np.sort(vals)
        except Exception:
            vals = np.zeros(0)

        res = np.zeros(k)
        res[:len(vals)] = vals
        return res.tolist()

    # --- WL hash (unchanged) ---
    def _c_mix(self, a, b):
        mask = 0xFFFFFFFF
        b &= mask
        a &= mask
        term1 = (b + 0x9e3779b9) & mask
        return ((a ^ term1) + ((a << 6) & mask) + ((a >> 2) & mask)) & mask

    def get_wl_signature(self, iterations=2):
        if self.num_nodes == 0:
            return 0
        labels = {n: (1 if self.G.in_degree(n) == 0 else 2) for n in self.G.nodes()}

        for _ in range(iterations):
            next_labels = labels.copy()
            for n in self.G.nodes():
                if self.G.in_degree(n) == 0:
                    continue
                cur_hash = labels[n]
                for pred in sorted(self.G.predecessors(n)):
                    p_label = labels[pred]
                    if self.G[pred][n].get('type') == 1:
                        p_label = self._c_mix(p_label, 0xDEADBEEF)
                    cur_hash = self._c_mix(cur_hash, p_label)
                next_labels[n] = cur_hash
            labels = next_labels

        roots = [n for n, d in self.G.out_degree() if d == 0]
        return labels[roots[0]] if roots else 0

    # -------------------------
    # New: cut-context features
    # -------------------------
    def _choose_root(self, roots):
        """
        cut 内可能出现多个 out_degree==0 的“根候选”，
        默认选全局 L 最大的那个（更像实际 cone 的输出端）。
        """
        if not roots:
            return None
        return max(roots, key=lambda n: self.L_global.get(n, 0))

    def get_cut_context_features(self):
        """
        你指定要的特征：
        - R(root)
        - SlackProxy(root)   (用 D-(L+R)，如需更贴近 required 可切换为 SlackProxy2)
        - max L(leaf_i)
        - max/mean/std Leaf_Fanout
        - frac_leaf_PI
        """
        if self.num_nodes == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        leaves = [n for n, d in self.G.in_degree() if d == 0]
        roots = [n for n, d in self.G.out_degree() if d == 0]
        root = self._choose_root(roots)

        # 1) R(root)
        R_root = float(self.R_global.get(root, 0)) if root is not None else 0.0

        # 2) SlackProxy(root)
        #    默认用 SlackProxy = D-(L+R)。如果你更想贴近 map 的 required 回传，可用 SlackProxy2_global。
        Slack_root = float(self.SlackProxy_global.get(root, 0)) if root is not None else 0.0
        # Slack_root = float(self.SlackProxy2_global.get(root, 0)) if root is not None else 0.0

        # 3) max L(leaf)
        if leaves:
            max_L_leaf = float(max(self.L_global.get(lf, 0) for lf in leaves))
        else:
            max_L_leaf = 0.0

        # 4) Leaf_Fanout stats
        leaf_fos = []
        for lf in leaves:
            fo_total = self.fanout_global.get(lf, 0)

            if self.leaf_fanout_mode == "external":
                # cut 内部从 leaf 指向的边数（只算 subg 内）
                fo_in_cut = int(self.G.out_degree(lf))
                fo_ext = fo_total - fo_in_cut
                if fo_ext < 0:
                    fo_ext = 0
                leaf_fos.append(fo_ext)
            else:
                leaf_fos.append(fo_total)

        if leaf_fos:
            a = np.asarray(leaf_fos, dtype=np.float64)
            max_leaf_fo = float(a.max())
            mean_leaf_fo = float(a.mean())
            std_leaf_fo = float(a.std(ddof=0))
        else:
            max_leaf_fo = mean_leaf_fo = std_leaf_fo = 0.0

        # 5) frac_leaf_PI
        if leaves:
            frac_leaf_pi = float(np.mean([1.0 if self.is_pi_global.get(lf, False) else 0.0 for lf in leaves]))
        else:
            frac_leaf_pi = 0.0

        return [R_root, Slack_root, max_L_leaf, max_leaf_fo, mean_leaf_fo, std_leaf_fo, frac_leaf_pi]

    # -------------------------
    # Main API
    # -------------------------
    def extract_all(self, subg):
        """
        subg: cut 子图，格式 {node: [(fanout, is_not_edge), ...]}

        返回：
          full_vector, names
        """
        self.raw_data = subg
        self.G = self._build_weighted_graph()

        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()

        # 新增上下文特征（7维）
        f_ctx = self.get_cut_context_features()

        # 原有特征
        f_time = self.get_timing_stats()               # [3]
        f_base = self.get_basic_stats()                # [1]
        f_stru = self.get_structural_stats(f_time[1])  # [2]
        f_inv  = self.get_inverter_stats()             # [2]
        f_cont = self.get_continuous_topology()        # [3]
        f_lap  = self.get_laplacian_eigenvalues(k=5)   # [5]
        wl_hash = self.get_wl_signature()              # [1]

        full_vector = f_ctx + f_time + f_base + f_stru + f_inv + f_cont + f_lap + [wl_hash]

        names = [
            "R_root", "SlackProxy_root", "Max_L_leaf",
            "Max_Leaf_Fanout", "Mean_Leaf_Fanout", "Std_Leaf_Fanout",
            "Frac_Leaf_PI",
            "Weighted_Depth", "Topo_Depth", "Inv_Penalty",
            "Node_Count",
            "Avg_Width", "Max_Internal_Fanout",
            "Inv_Ratio", "Root_Inv_Count",
            "Chain_Density", "Fork_Density", "Avg_Curvature",
            "Lap_Eig1", "Lap_Eig2", "Lap_Eig3", "Lap_Eig4", "Lap_Eig5",
            "WL_Hash"
        ]

        return full_vector, names
