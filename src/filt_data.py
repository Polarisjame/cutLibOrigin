import os
import torch
import random

def filt_data(data_path, exec_limit):
    # 先判断data_path文件大小是否超过30GB
    file_size = os.path.getsize(data_path)
    if file_size > exec_limit * 1024 * 1024 * 1024:
        print(f"File size {file_size} exceeds {exec_limit}GB, start processing.")
    else:
        print(f"File size {file_size} is within limit, skipping.")
        return
    data = torch.load(data_path)
    
    original_len = len(data)
    
    # 使用列表推导式保留符合条件的数据
    data = [block for block in data if random.randint(0, 10) < 6]
    
    new_len = len(data)
    print(f"Reduced elements from {original_len} to {new_len}")

    torch.save(data, data_path)
    new_size = os.path.getsize(data_path)
    print(f"Original size: {file_size}, New size: {new_size}")

for design in os.listdir("/data/zhoulingfeng/data_cutLib/save_subgraph/save_designs_full_b16"):
    if not design.endswith(".pt"):
        continue
    print(f"Processing design: {design}")
    data_path = os.path.join("/data/zhoulingfeng/data_cutLib/save_subgraph/save_designs_full_b16", design)
    filt_data(data_path, exec_limit=40)