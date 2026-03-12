import os
import dgl
import torch
import random
import numpy as np
import gc
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
import time
import sys
from datetime import datetime
import builtins

from models.attn_sage import NeighborSampler
from data_preprocess.data_process import parse_aig2dg
from subutils.utils import seed_everything
from subutils.config import get_opt

# 保存原始的 print 函数
original_print = builtins.print

# 重写 print 函数
def new_print(*args, **kwargs):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    original_print(f"[{timestamp}] ", end="")
    original_print(*args, **kwargs)

# 替换全局的 print 函数
builtins.print = new_print

train_design = ['spi_orig', 'i2c_orig', 'wb_dma_orig', 'simple_spi_orig', 'pci_orig', 'wb_conmax_orig', 'ethernet_orig', 'vga_lcd_orig', 'bp_be_orig', 'jpeg_orig', 'idft_orig', 'dft_orig', 'sha256_orig', 'aes_secworks_orig', 'aes_xcrypt_orig', 'tinyRocket_orig', 'picosoc_orig', 'dynamic_node_orig']
valset = ['ss_pcm_orig', 'usb_phy_orig', 'sasc_orig', 'ac97_ctrl_orig', 'mem_ctrl_orig', 'des3_area_orig', 'aes_orig', 'fir_orig', 'iir_orig', 'tv80_orig', 'fpu_orig']

def extract_dataset(res_list, use_batchs, labels, cell_fanouts, dgl_graph, design):
    rea = 0
    for ind,x in enumerate(use_batchs):
        if ind % 10 == 0:
            print(f"{ind}/{len(use_batchs)}")
        label = labels[x,:]
        blocks = sampler.sample_blocks(x)

        block_dict = {'block': blocks, 'feats':blocks[-1].dstdata['feats']['cell'], 'design':design, 'lable': label}
        
        res_list.append(block_dict) 

if __name__ == '__main__':
    args = get_opt()
    seed_everything(42)

    graphset = torch.load(os.path.join(args.dataset_dir,f'graph_with_dg_cell2cell_{args.case}.pt'))

    for graph in graphset:
        res_list = []
        design = graph['design']
        if design in ['hyp_orig']:
            continue
        target_out_dir = f'{args.dataset_dir}/save_designs_b{args.batch_size}/{args.case}/blockset_{design}_hop{args.n_layer}_batch{args.batch_size}.pt'
        if os.path.exists(target_out_dir):
            print(f"File already exists, skipping {design}", flush=True)
            continue
        os.makedirs(os.path.dirname(target_out_dir), exist_ok=True)
        print(design, flush=True)
        labels = graph['labels']
        # gate_gts = graph['gate_gts']
        cell_fanouts = graph['cell_fanouts']
        dgl_graph:dgl.DGLGraph = graph['graph']
        cell_ids = torch.arange(0, dgl_graph.num_nodes(ntype='cell'))
        real_cells = []
        real_labels = []
        for cell_id,label in zip(cell_ids, labels):
            if label[2] > 0:
                real_cells.append(cell_id)
                real_labels.append(label)
        cell_fanouts = torch.tensor(cell_fanouts,dtype=torch.int,requires_grad=False)

        sampler = NeighborSampler(dgl_graph, args.n_layer, 'cuda:0')
        dataloader = DataLoader(real_cells, batch_size=args.batch_size, shuffle=True)
        batchs = dataloader
            
        check = time.time()
        # print(f"Sample Label Elapse:{check-start}", flush=True)
        extract_dataset(res_list, batchs, labels, cell_fanouts, dgl_graph, design)
        print(f"Done Design {design}")
        torch.save(res_list, target_out_dir)
        del res_list
        gc.collect()

    print("finished")




# dataloader = DataLoader()
