from cffi import FFI
import numpy as np

ffi = FFI()
ffi.cdef("unsigned Abc_TtCanonicize(uint64_t *pTruth, int nVars, char *pCanonPerm);")
ffi.cdef("void Abc_TtNormalizeSmallTruth(uint64_t * pTruth, int nVars);")
lib = ffi.dlopen("./abc_so/libabc.so")

nVars = 3                         # 固定变量数
pTruth = ffi.new("uint64_t[]", 1)   # 只分配一次
perm   = ffi.new("char[]", nVars)
def pack_truth(bits):
    word = 0
    for idx, val in enumerate(bits):
        if val:
            word |= (1 << idx)
    buf = np.array([word], dtype=np.uint64)
    ffi.memmove(pTruth, ffi.from_buffer(buf), buf.nbytes)
    lib.Abc_TtNormalizeSmallTruth(pTruth, nVars)

def canonize():
    phase = lib.Abc_TtCanonicize(pTruth, nVars, perm)
    return phase, [perm[i] for i in range(nVars)], int(pTruth[0])

pack_truth([0,0,0,0,0,0,0,1])
print("Original truth table: ", int(pTruth[0]))
phase1, perm1, tt1 = canonize()
tt1 = pTruth[0]
pack_truth([0,0,0,0,0,0,1,0])
print("Original truth table: ", int(pTruth[0]))
phase2, perm2, tt2 = canonize()
tt2 = pTruth[0]


print(tt1, tt2)