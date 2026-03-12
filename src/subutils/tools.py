from datetime import datetime
import logging
from torch_geometric.data import Data
import torch

class Log:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath,'w')
        self.file.truncate()
        self.file.close()
        logging.basicConfig(filename=filepath, 
                        level=logging.INFO,  # 设置日志级别
                        format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()
    def __call__(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)
        # print(datetime.now(), *args, **kwargs, file=self.file, flush=True)

class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" :
            return 1
        else:
            return 0

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class R2Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.true_list = []
        self.pred_list = []

    def update(self, y_true, y_pred):
        self.true_list+=y_true.cpu().detach().tolist()
        self.pred_list+=y_pred.cpu().detach().tolist()

    def getVal(self):
        y_true = torch.tensor(self.true_list)
        y_pred = torch.tensor(self.pred_list)
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2).numpy()
        ss_residual = torch.sum((y_true - y_pred) ** 2).numpy()
        return 1 - (ss_residual / ss_total)
    

# 计算 MAPE
def calculate_mape(y_true, y_pred):
    return torch.sum(torch.abs((y_true - y_pred) / y_true)).numpy() * 100
