
import torch
from tqdm import tqdm
from utils.utils import log_and_print
from loss import attention_loss, cal_spearmanr_rl2
def run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler):
    backbone, neck, head = network
    rho_best, epoch_best, rl2_best = 0, 0, 0
    for epoch in range(cfg.epoch_num):
        for split in ['train','test']:
            true_scores = []
            pred_scores = []
            
            if split == 'train':
                backbone.train()
                head.train()
                neck.train()
                torch.set_grad_enabled(True)
            else:
                backbone.eval()
                head.eval()
                neck.eval()
                torch.set_grad_enabled(False)
                
        self_map_lst = []
        cross_map_lst = []
        losses = 0
        #import pdb; pdb.set_trace()
        for data_ in tqdm(data_loaders[split]):
            data, clip_info = data_

            score = data["score"].float().cuda()
            video = data["video"]
            
            if cfg.split_feats:
                bs, frame, feats  = video.shape
                if "feats" in data:
                    clip_feats = data["feats"].cuda()
                else:
                    video = video.reshape(video.shape[0],3,48,16,224,224).cuda()
                    clip_feats = torch.empty(bs, video.shape[2], feats).cuda()
                    for i in range(frame):
                        clip_feats[:,i] = backbone(video[:,:,i,:,:,:].cuda())[1].squeeze(-1).squeeze(-1).squeeze(-1)
            else:
                bs, frame, h, w  = video.shape
                clip_feats = backbone(video)[1].squeeze(-1).squeeze(-1).permute(0,2,1)
            
            tgt_weight, graph_attn = neck(clip_feats)
            probs, weight, means, var = head(tgt_weight)
            
            pred_scores.extend([i.item() for i in probs])
            true_scores.extend(score.cpu().numpy())
            
            kld_loss, self_map_lst, cross_map_lst = attention_loss(graph_attn, kld, self_map_lst, cross_map_lst)
            mes_loss = mse(probs,score)
            
            if cfg.att_loss:
                    loss = mes_loss + kld_loss
            else:
                loss = mes_loss
                
            losses += loss
            rho, p, rl2 = cal_spearmanr_rl2(pred_scores, true_scores)
            
            
            if split=="train":
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("当前学习率:", param_group['lr'])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # show information
        log_and_print(base_logger, f'Epoch: {epoch}, {split} correlation: {rho}')
        
        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            rl2_best = rl2
            log_and_print(base_logger, '-----New best found!-----')
            torch.save({'epoch': epoch,
                            'backbone': backbone.state_dict(),
                            'neck': neck.state_dict(),
                            'head': head.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best}, f'ckpts/{cfg.dataset_name}_{cfg.seed}_{cfg.gym_label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}.pt') 
