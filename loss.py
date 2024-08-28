import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats


def attention_loss(graph_attn, kld, self_map_lst, cross_map_lst):
    kld_loss = 0
    self_maps, cross_maps, memorys = graph_attn
    for self_map, cross_map, memory in zip(self_maps, cross_maps, memorys):
        # self attention map
        G_D = torch.matmul(self_map.transpose(0,1), self_map.transpose(0,1).transpose(-1,-2))
        G_D = torch.div(G_D,self_map.size(-1))
        soft_gd_self = F.softmax(G_D,dim=1)

        # cross attention map
        G_E = torch.matmul(cross_map.transpose(0,1), cross_map.transpose(0,1).transpose(-1,-2))
        G_E = torch.div(G_E,cross_map.size(-1))
        soft_gd_cross = F.softmax(G_E,dim=1)
       
        kld_loss +=  kld(F.log_softmax(G_D, dim=-1), F.softmax(G_E, dim=-1))

    for i in range(len(soft_gd_cross)):
        self_map_lst.append(soft_gd_self[i])
        cross_map_lst.append(soft_gd_cross[i])
        
    return kld_loss, self_map_lst, cross_map_lst

def cal_spearmanr_rl2(pred_scores, true_scores):
    # calculate spearman correlation
    rho, p = stats.spearmanr(pred_scores, true_scores)
    # calculate rl2
    pred_scores_ = np.array(pred_scores)
    true_scores_ = np.array(true_scores)
    rl2 = np.power((pred_scores_ - true_scores_) / (true_scores_.max() - true_scores_.min()), 2).sum() / true_scores_.shape[0]
    return rho, p, rl2