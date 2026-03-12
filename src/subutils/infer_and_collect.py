import os
import re
import threading
import torch
from torch.optim import SGD, lr_scheduler
from torch.nn import MSELoss
import dgl
import numpy as np
import random
import time

from models.CellPool_wo_lib import CellPool
from subutils.tools import Log, AverageMeter, R2Meter, calculate_mape
from models.attn_sage import NeighborSampler

class Dataloader():
    def __init__(self, dataset_dir, log):
        self.dataset_dir = dataset_dir
        self.data = {}
        self.lock = threading.Lock()
        self.log = log

    def load_data(self, design_file):
        """加载指定设计的数据并存储"""
        data = torch.load(os.path.join(self.dataset_dir, design_file))
        with self.lock:
            self.data[design_file] = data
        self.log(f"Data loaded for {design_file}")
    
    def get_data(self, design_file):
        """获取已加载的设计数据"""
        with self.lock:
            return self.data.get(design_file, None)

    def del_data(self, design_file):
        """删除指定设计的数据"""
        with self.lock:
            if design_file in self.data:
                del self.data[design_file]
                self.log(f"Data deleted for {design_file}")
            else:
                self.log(f"No data found for {design_file}")

class Trainer():
    
    def __init__(self, model:CellPool, args, device, dataset_dir, logger=None) -> None:
        self.train_epochs = args.epochs
        self.data_root = args.data_root
        self.device = device
        self.batch_size = args.batch_size
        self.num_worker = args.num_worker
        self.infeat = args.in_feat
        if logger is None:
            self.logger = Log(f'./log/deepPool_lr{self.lr}_batch{self.batch_size}.log')
        else:
            self.logger = logger

        model.cuda()
        self.lib_min = 0.7
        self.lib_max = 7.0
        self.delay_mean = 157.21139098502647#181.75262794900135
        self.delay_std = 189.58626306996536#199.15174363746996
        self.delay_max = 1440.192444#12588.473
        self.delay_min =5.638098
        self.delay_norm_min = -0.7994951244388919 - 3e-6#-0.8843238062208247 - 1e-6

        self.l2_theta = 0.005
        self.model = model
        self.model_name = args.model_name
        self.dataset_dir = dataset_dir

        self.design_file_list = os.listdir(self.dataset_dir)
        self.data_loader = Dataloader(self.dataset_dir, self.logger)
        self.current_design_idx = 0
        self.loading_thread = None

        self.device = device
        self.use_normal = args.use_normal
        
        self.model_save = args.model_save
        self.num_hops = args.n_layer
        self.trainset_rate = args.trainset_rate
        self.lr_gnn = args.lr_gnn
        self.log_step = args.log_step
        self.save_root = f'/data/zhoulingfeng/data_cutLib/save_infer'
        self.keep_last_collected = getattr(args, "keep_last_collected", 2)
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        base_param = []
        for pname, p in self.model.gnnmodel.named_parameters():
            if 'hop' in pname or 'pearson' in pname or 'fanout_rate' in pname or 'lib_thresh' in pname:
                continue
            base_param += [p]

        self.optimizer = SGD([
                {"params":self.model.hop, "lr": args.lr_hop},
                {"params":base_param, "lr": args.lr_gnn},
                {"params":self.model.fanout_rate, "lr":args.lr_fanout},
                {"params":self.model.deepgate.parameters(), "lr": args.lr_dg},
                {"params":self.model.projection.parameters(), "lr": args.lr_gnn}] , momentum=0.9, weight_decay=args.l2_decacy)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(3e4), T_mult=2)
        assert args.loss_reduction in ['sum', 'mean'], f"Error Reduction: {args.loss_reduction}"
        self.loss_reduction = args.loss_reduction
        self.lossF = MSELoss(reduction=args.loss_reduction)

    def _prune_old_collected_results(self, keep: int = 2) -> None:
        if keep is None or keep <= 0:
            return

        log = self.logger
        try:
            collected_files = []
            for name in os.listdir(self.save_root):
                match = re.match(r"^collected_step(\d+)\.pt$", name)
                if match is None:
                    continue
                collected_files.append((int(match.group(1)), os.path.join(self.save_root, name)))

            if len(collected_files) <= keep:
                return

            collected_files.sort(key=lambda x: x[0])
            for _, path in collected_files[:-keep]:
                try:
                    os.remove(path)
                    log(f"Removed old collected_result: {path}")
                except FileNotFoundError:
                    continue
                except Exception as e:
                    log(f"Failed to remove old collected_result {path}: {e}")
        except Exception as e:
            log(f"Prune collected_result failed: {e}")

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        model_ckpt = ckpt['checkpoint']
        self.model.load_state_dict(model_ckpt,strict=True)
        self.logger(f"load ckpt:{ckpt_path} Done")
    
    def normalize_label(self, delay):
        delay_label = delay.unsqueeze(1)
        if self.use_normal:
            delay_norm = (delay_label - self.delay_mean) / self.delay_std - self.delay_norm_min
            return delay_norm
        else:
            return delay
    def unnormalize_label(self, delay):
        delay_label = delay
        if self.use_normal:
            delay_unnorm = (delay_label + self.delay_norm_min) * self.delay_std + self.delay_mean
            return delay_unnorm
        else:
            return delay
    
    def save_model(self, epoch, step):
        path = f"/data/zhoulingfeng/checkpoints/{self.model_name}_batch{self.batch_size}_epoch{epoch}_step{step}.pth"
        torch.save({
            'epoch': epoch,
            'checkpoint': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    

    def inferAndCollect(self):
        log = self.logger
        train_mape_states = AverageMeter()
        train_r2_states = R2Meter()
        train_loss_states = AverageMeter()
        self.model.to(self.device)
        # valset = self.graph_set
        hop_set = []
        collected_result = []
        self.current_design_idx = 0
        self.loading_thread = threading.Thread(target=self.data_loader.load_data, args=(self.design_file_list[self.current_design_idx],))
        self.loading_thread.start()
        step = 0   
        train_loss_states.reset()
        train_r2_states.reset()
        train_mape_states.reset()
        with torch.no_grad():
            while self.current_design_idx < len(self.design_file_list):
                self.model.eval()
                if self.loading_thread is not None:
                    self.loading_thread.join()
                design_file = self.data_loader.get_data(self.design_file_list[self.current_design_idx])
                next_design_idx = self.current_design_idx + 1
                if next_design_idx < len(self.design_file_list):
                    next_design = self.design_file_list[next_design_idx]
                    self.loading_thread = threading.Thread(target=self.data_loader.load_data, args=(next_design,))
                    self.loading_thread.start()

                self.current_design_idx += 1
                
                design = design_file[0]['design']
                dataset = design_file
                random.shuffle(dataset)
                for blockset in dataset:
                    blocks = blockset['blocks']
                    design = blockset['design']
                    if len(blocks[-1].edges(etype='mapper')[-1].unique()) != blocks[-1].num_dst_nodes():
                        continue
                    if design in ['hyp_orig']:
                        continue
                    step += 1
                    if step >= 6680 and step <= 6700:
                        continue
                    block_hf = blockset['hf']
                    block_hs = blockset['hs']
                    label = blockset['lable'][:,2]
                    b = label.shape[0]
                    if b == 1:
                        continue
                    cell_fanout = blockset['cell_fanout']
                    root_index = blockset['root_index']
                    root_index = torch.cat(root_index).view(-1) 
                    neighbor_dict = blockset['neighbor_dict']
                    label = self.normalize_label(label)
                    block_hs.to(self.device)
                    block_hf.to(self.device)
                    try:
                        hs_root = block_hs[root_index]
                        hf_root = block_hf[root_index]
                    except Exception as e:
                        log(block_hs, block_hs.shape, root_index, f"Error at design {design}, step {step}: {e}")
                        exit(-1)
                    hf_root = torch.cat((hf_root,hs_root),dim=-1)
                    cell_fanout = torch.tensor(cell_fanout, dtype=torch.int, requires_grad=False).view(-1,1).to(self.device)
                    hf_cell = self.model.cell_pooling(blocks, hf_root, self.infeat, cell_fanout, neighbor_dict)
                    del block_hs, block_hf, hs_root
                    hf_root = hf_root.to(self.device)

                    batch_inputs = {'aig':hf_root,'cell':hf_cell.to(self.device)}
                    delay_label = label.to(self.device)

                    pred_res = self.model(blocks, batch_inputs)
                    # stage1 = torch.cuda.memory_allocated(0)/8/1024/1024

                    # loss = self.lossF(pred_res, delay_label)
                    delay_label = delay_label.view(-1).detach().cpu()
                    pred_res = pred_res.view(-1).detach().cpu()
                    if step % self.log_step == 0:
                        log(f"Design[{design}] | Step[{step}]")
                    
                    delay_label = [a if a > 0 else b for a,b in zip(delay_label, pred_res)]
                    
                    # print(blocks, blocks[-1], blocks[-1].ndata, blocks[-1].ndata['semi'], sep='\n')
                    feats = blocks[-1].ndata['feats']['cell']
                    indexes = blocks[-1].ndata['index']['cell']
                    for label, feat, index in zip(delay_label, feats, indexes):
                        # index: [cut_rt, i]
                        label = label.view(-1,1)
                        collected_result.append((self.unnormalize_label(label).detach().cpu(), feat.detach().cpu(), index.detach().cpu(), design))
                    torch.cuda.empty_cache()
                
                # 保存collected_result
                torch.save(collected_result, f"{self.save_root}/collected_step{step}.pt")
                self._prune_old_collected_results(keep=self.keep_last_collected)
                log(f"Saved collected results, design {design}")
                self.data_loader.del_data(self.design_file_list[self.current_design_idx-1])
    
