import numpy as np
import random
import numpy as np
import torch

def pack_truth(tt_num, pTruth, nVars, ffi, lib):
    buf = np.array([tt_num], dtype=np.uint64)
    ffi.memmove(pTruth, ffi.from_buffer(buf), buf.nbytes)
    lib.Abc_TtNormalizeSmallTruth(pTruth, nVars)

def canonize(tt_num, pTruth, nVars, ffi, lib):
    perm   = ffi.new("char[]", nVars)
    pack_truth(tt_num, pTruth, nVars, ffi, lib)
    phase = lib.Abc_TtCanonicize(pTruth, nVars, perm)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False