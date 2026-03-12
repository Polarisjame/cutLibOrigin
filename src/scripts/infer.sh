#! /bin/bash
export CUDA_VISIBLE_DEVICES=2
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------DeepCut Training Shell Script--------------------

data_root='/data/zhoulingfeng/data/save_json' # unprocessed data
dataset_dir='/data/zhoulingfeng/data_cutLib/save_subgraph' # nn.Dataset save
log_dir='./log/train_log_DeepCutDelay_infer.log'
model_name=DeepCutDelay_agg_rand_cell2cell_fanout_v2
random_seed=42
model_save=20000

# pretrain
dg_pretrained_path='./deepgate/pretrained/model.pth'
model_ckpt='.'
model_ckpt='/data/zhoulingfeng/checkpoints/DeepCutDelay_agg_rand_cell2cell_fanout_v2_ddp_batch16_epoch16_step9992.pth'

# dataloader
batch_size=16
num_worker=0
# log_step=$((200 / batch_size))
log_step=10
lr_dg=8e-3
lr_hop=5e-3
lr_gnn=7e-4
lr_fanout=3e-4
lr_pearson=5e-3
l2_decacy=1e-5
epochs=30
trainset_rate=0.7
loss_beta=5
loss_reduction='mean' # sum mean

# model
n_layer=3
drop_rate=0.1
num_heads=8
lamma2_multi=8
gamma=0.4
num_hop=1
slope=0.1
in_feat=256
out_feat=256
hidden_dim=256
mlp_hidden_dim=256
activate='prelu' #prelu relu sigmoid
diag=0.7
fanout_rate=0.6
lib_thresh=0.5

if true; then

    nohup python3 -u ./src/main.py \
        --bias\
        --use_cuda\
        --use_normal\
        --lib_thresh ${lib_thresh}\
        --model_ckpt ${model_ckpt}\
        --fanout_rate ${fanout_rate}\
        --lamma2_multi ${lamma2_multi}\
        --lr_fanout ${lr_fanout}\
        --loss_beta ${loss_beta}\
        --loss_reduction ${loss_reduction}\
        --data_root ${data_root}\
        --log_dir ${log_dir}\
        --log_step ${log_step}\
        --dg_pretrained_path ${dg_pretrained_path}\
        --dataset_dir ${dataset_dir}\
        --random_seed ${random_seed}\
        --batch_size ${batch_size}\
        --num_worker ${num_worker}\
        --diag ${diag}\
        --lr_dg ${lr_dg}\
        --lr_hop ${lr_hop}\
        --lr_gnn ${lr_gnn}\
        --l2_decacy ${l2_decacy}\
        --epochs ${epochs}\
        --trainset_rate ${trainset_rate}\
        --n_layer ${n_layer}\
        --drop_rate ${drop_rate}\
        --num_heads ${num_heads}\
        --gamma ${gamma}\
        --num_hop ${num_hop}\
        --slope ${slope}\
        --in_feat ${in_feat}\
        --out_feat ${out_feat}\
        --hidden_dim ${hidden_dim}\
        --mlp_hidden_dim ${mlp_hidden_dim}\
        --activate ${activate}\
        --model_save ${model_save}\
        --model_name ${model_name}\
    >log/erroroutDeepCutDelay_infer.log 2>&1 &
fi

