import argparse
import yaml
import json
import random, os, torch
import numpy as np

from subutils.infer_and_collect import Trainer
from subutils.config import get_opt
from subutils.tools import Log
from models.CellPool_wo_lib import CellPool


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

def main(opt, log):
    
    dataset_dir = os.path.join(opt.dataset_dir,f'save_designs_full_b{opt.batch_size}')
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = CellPool(opt, device=device, activate=opt.activate)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total_params}")
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total Trainable parameters: {total_trainable_params}")

    trainer = Trainer(model, opt, device, dataset_dir, log)
    if opt.model_ckpt != '.':
        trainer.load_checkpoint(opt.model_ckpt)
    # Train
    
    with torch.no_grad():
        trainer.inferAndCollect()

if __name__ == "__main__":
    opt = get_opt()
    args_dict = vars(opt)
    with open("params.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    log = Log(opt.log_dir)
    log(f'processId: {os.getpid()}')
    log(f'prarent processId: {os.getppid()}')
    log(json.dumps(opt.__dict__, indent=4))
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_torch(opt.random_seed)
    
    main(opt, log)