#! /bin/bash
export CUDA_VISIBLE_DEVICES=2

data_path='/data3/zhoulingfeng/data_cutLib/save_infer_gt/collected_step31428.pt'
# dataset_path='/data/zhoulingfeng/data_cutLib/save_subgraph/save_designs_b16'
dataset_path='/home/zhoulingfeng/data/cutLibData'
K=115
latent_dim=8
min_cluster=100
train_ratio=0.8
k_min=50

python3 src/entry_script/extract_cutLib.py \
    --data_path $data_path \
    --dataset_path $dataset_path \
    --K $K \
    --k_min $k_min \
    --latent_dim $latent_dim \
    --min_cluster $min_cluster \
    --train_ratio $train_ratio \
    --eval_pls_effect \
    --split_method inside_cluster # 'inside_cluster', 'across_cluster'
