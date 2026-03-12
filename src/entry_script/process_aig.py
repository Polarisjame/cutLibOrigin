import os
import subprocess
import threading
from subutils.Parser import AigParser, Aig2MapParser, DelayParser
import sys

print(f"Cur PID: {os.getpid()}")

dataset = sys.argv[1] if len(sys.argv) > 1 else 'openABC'
dataset_aig_path = f'/home/zhoulingfeng/data/open-sourceDataset/{dataset}'
# dataset_aig_path = '/home/zhoulingfeng/data/open-sourceDataset/OpenABC/aig_strashed'
data_root = '/home/zhoulingfeng/data/cutLibData'
if not os.path.exists(dataset_aig_path):
    os.mkdir(dataset_aig_path)
data_save_dir = f'{data_root}/save_json_{dataset}'
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir, exist_ok=True)

# # Bench Parser
# bench_parser = BenchProjParser(dataset, dataset_root, 'bench_openabcd')
# bench_parser.process()

aig_file_parser = AigParser(f'{data_root}/{dataset}', './log/abc_cmd.log')
aigmapParser = Aig2MapParser(dataset, './log/abcmap_exec.log')
delayParser = DelayParser(f'{data_root}/{dataset}','./log/delayLogger.log')

# AigParser
cnt = 0
for aig_file in os.listdir(dataset_aig_path):
    if not aig_file.endswith('.aig'):
        cnt += 1
        print(f"Skip {aig_file} as it is not an AIG file.")
        continue

    aig_file_path = os.path.join(dataset_aig_path,aig_file)
    design_name = aig_file_path.split('/')[-1].split('.')[0]
    design_save_dir = os.path.join(data_save_dir,design_name)
    if os.path.exists(f'{design_save_dir}/data_2.json'):
            print(f"{design_name} has been executed before")
            cnt += 1
            # delayParser.process(design_name,data_save_dir)
            continue
    aig_file_parser.log(aig_file_path)
    aig_file_parser.process(aig_file_path)
    aigmapParser.process(aig_file_path, data_save_dir)
    delayParser.process(design_name,data_save_dir)

    cnt += 1
    print(f"Finished: {cnt}/{len(os.listdir(dataset_aig_path))}")
    