import torch
from torch import nn
import copy
from einops import rearrange
from torch.nn.functional import softmax
from torch.nn import LeakyReLU
from torch import pow

def process_mask(relative_topo_d, gamma, hop, slope=0.1):
    # hop:num_head, num_neigh
    activ = LeakyReLU(negative_slope=slope)
    relative_topo_d = relative_topo_d.unsqueeze(1)
    r_d = relative_topo_d-hop
    r_d = activ(r_d)
    r_m = pow(gamma,r_d)
    return r_m.unsqueeze(-2)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SiLUMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.1):
        """
        Initialize a SiLU-based MLP.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output features.
            num_layers (int): Number of layers in the MLP.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(SiLUMLP, self).__init__()
        layers = []

        # Input to hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_prob))

        # Hidden to output
        layers.append(nn.Linear(hidden_dim, output_dim))

        # Combine layers
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_head = num_head
        self.scaling = (d_model // self.num_head) ** -0.5  # emb_size = embeddingsize * num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.out_projection = SiLUMLP(d_model, 768, d_model)

    def forward(self, q, k, v, mask_head=None, pearson_matrix=None, printout = False):
        queries, keys, values = [rearrange(l(x), "b n (h d) ->b h n d", h=self.num_head) for l,x in zip(self.linears,(q,k,v))]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        energy *= self.scaling
        att = softmax(energy, dim=-1)
        # if printout:
        #     b, h, q, k = att.shape
        #     with open('./file_test.log', 'a') as f:
        #         f.write('attn_orig\n')
        #         att_list = att.tolist()
        #         for i in range(b):
        #             for j in range(h):
        #                 for k in range(q):
        #                     f.write(str(att_list[i][j][k]) + '; ')
        #             f.write('\n')
        if mask_head is not None:
            # print('S Shape Mask: ', energy.shape, mask_head.shape)
            att = att*mask_head
        
        if pearson_matrix is not None:
            att = torch.einsum('bhqk, bhkk -> bhqk', att, pearson_matrix)

        # if printout:
        #     b, h, q, k = att.shape
        #     with open('./file_test.log', 'a') as f:
        #         f.write('attn_mask\n')
        #         att_list = att.tolist()
        #         for i in range(b):
        #             for j in range(h):
        #                 for k in range(q):
        #                     f.write(str(att_list[i][j][k]) + '; ')
        #             f.write('\n')
                    

        agg_feat = torch.einsum('bhal, bhlv -> bhav ', att, values)
        agg_feat = rearrange(agg_feat, "b h n d -> b n (h d)")
        return agg_feat

class ThresholdMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold):
        ctx.save_for_backward(scores, threshold)
        return (scores > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        scores, threshold = ctx.saved_tensors
        sigmoid_gate = torch.sigmoid(scores - threshold)
        grad_approx = sigmoid_gate * (1 - sigmoid_gate)  # 梯度近似
        
        # 计算阈值梯度（聚合所有位置的贡献）
        grad_threshold = -grad_approx * grad_output
        grad_threshold = grad_threshold.sum()
        
        # 计算scores梯度（传递给Q和K）
        grad_scores = grad_approx * grad_output
        
        return grad_scores, grad_threshold


class PointerNetwork(nn.Module):
    def __init__(self, d_model, choose_thresh, device):
        """
        Initialize a Pointer Network.
        Args:
            choose_thresh (tensor): Threshold for choosing attention scores, [1].
        """
        super(PointerNetwork, self).__init__()
        self.scaling = d_model  ** -0.5  
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.out_project = SiLUMLP(d_model, 768, d_model)
        self.choose_thresh = choose_thresh
        self.device = device
        self.activate = nn.ReLU()

    def hard_threshold(self, score, threshold):
        return self.activate(score - threshold)
    
    def forward(self, q, k, v):
        """_summary_

        Args:
            q (Tensor): m, d
            k,v (_type_): n, d
        """
        queries, keys, values = [l(x) for l,x in zip(self.linears,(q,k,v))]
        attn_score = torch.einsum('bqd, bkd -> bqk', queries, keys)  # b, 1, n
        attn_score *= self.scaling

        # 将attn_score中大于choose_thresh的位置设为1，小于choose_thresh的位置设为0，将对应values的值做maxpooling
        mask = ThresholdMask.apply(attn_score, self.choose_thresh)
        # 调整mask形状并转换为布尔类型
        mask = mask.permute(0, 2, 1).bool()  # 变为[b, k, 1]
        # 将未选中的特征填充为负无穷
        masked_features = values.masked_fill(~mask, float('-inf'))
        # 沿k维度取最大值
        output = torch.max(masked_features, dim=1)[0]
        output = torch.nan_to_num(output, neginf=0)

        lib_feat = self.out_project(output)
        
        return lib_feat, mask
