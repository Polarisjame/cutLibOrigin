import argparse

def get_opt():
    parser = argparse.ArgumentParser(description="Deeppool")
    
    # Data
    parser.add_argument("--data_root",type=str,default=r"/home/zhoulingfeng/data/cutLibData/save_json")
    parser.add_argument("--dataset_dir",type=str,default=r"/data/zhoulingfeng/data_cutLib/save_subgraph")
    parser.add_argument("--case",type=str,default=r"openABC")
    parser.add_argument('--local-rank', type=int, help='Local rank of the current process', default=0)
    

    # Model
    parser.add_argument("-l","--n_layer",type=int,help="num of GNN layers", default=3)
    parser.add_argument("--bias",action='store_true', help='use bias in GNN fc')
    parser.add_argument("--use_cuda",action='store_true', help='use cuda in DeepGate Pre Embedding')
    parser.add_argument("--in_feat",type=int,help="num of GNN input embedding", default=256)
    parser.add_argument("--out_feat",type=int,help="num of GNN output embedding", default=256)
    parser.add_argument("--hidden_dim",type=int,help="num of GNN hidden size", default=512)
    parser.add_argument("--drop_rate",type=float,help="GNN Dropout rate, default = 0.2", default=0.2)
    parser.add_argument("--num_heads",type=int,help="Head Num in MHA", default=4)
    parser.add_argument('-g',"--gamma",type=float,help="Attn Mask Gamma", default=0.7)
    parser.add_argument("--num_hop",type=float,help="Attn Mask Hop Init", default=1)
    parser.add_argument("--slope",type=float,help="Attn Mask LeakyRelu Param", default=0.1)
    parser.add_argument("--mlp_hidden_dim",type=int,help="MLP hidden dim", default=32)
    parser.add_argument("--activate",type=str,help="activation func", default='prelu')
    parser.add_argument("--diag",type=float,help="Init Corr Matrix Diag Val", default=0.7)
    parser.add_argument("--fanout_rate",type=float,help="Init CellFanout Rate", default=0.05)
    parser.add_argument("--lamma2_multi",type=int,help="Lamma2Multiple", default=8)
    parser.add_argument("--lib_thresh",type=float,help="Liberty Attn Threshold", default=0.35)

    # train
    parser.add_argument("-b","--batch_size",type=int, default=16)
    parser.add_argument("-n","--num_worker",type=int, default=0)
    parser.add_argument("--dg_pretrained_path",type=str,default=r"./deepgate/pretrained/model.pth")
    parser.add_argument("--model_ckpt",type=str,default='.')
    parser.add_argument("--lr_dg",type=float,default=100)
    parser.add_argument("--lr_gnn",type=float,default=1e-3)
    parser.add_argument("--lr_hop",type=float,default=1e-3)
    parser.add_argument("--lr_pearson",type=float,default=5e-4)
    parser.add_argument("--lr_fanout",type=float,default=8e-3)
    parser.add_argument("--l2_decacy",type=float,default=1e-5)
    parser.add_argument("--epochs",type=int, default=20)
    parser.add_argument("--trainset_rate",type=float, default=0.7)
    parser.add_argument("--random_seed",type=int, default=42)
    parser.add_argument("--loss_beta",type=float, default=1)
    parser.add_argument("--loss_reduction",type=str, default='mean')
    parser.add_argument("--model_save",type=int, default=5000)
    parser.add_argument("--model_name",type=str, default='DeepPooldebug')
    parser.add_argument("--use_normal",action='store_true', help='use MinMax Norm for label')

    
    # Log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default="./log/debug.log")
    args = parser.parse_args()
    return args