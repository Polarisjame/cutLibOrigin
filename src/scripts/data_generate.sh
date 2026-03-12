# !/bin/bash

CASE='forgeEDA'

data_root="/home/zhoulingfeng/data/cutLibData/save_json_${CASE}"
dataset_dir="/data/zhoulingfeng/data_cutLib/save_subgraph"
n_layer=3
batch_size=16

# echo $data_root

# 串行执行三个python脚本
nohup python3 -u src/entry_script/process_aig.py $CASE > ./log/process_aig_${CASE}.log 2>&1 &

# nohup python3 -u src/entry_script/sub_graph_extract.py --data_root $data_root --dataset_dir $dataset_dir --case $CASE > ./log/sub_graph_extract_${CASE}.log 2>&1 &

# nohup python3 -u src/entry_script/get_label_samp.py \
#         --n_layer ${n_layer}\
#         --batch_size ${batch_size}\
#         --num_worker 15\
#         --dataset_dir $dataset_dir --case $CASE > ./log/get_label_samp_${CASE}.log 2>&1 &