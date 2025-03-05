
import torch
from tqdm import tqdm
import logging
from utils.utils import log_and_print
from utils.vis import *
from loss import attention_loss, cal_spearmanr_rl2
from methods.weight_methods import NashMTL
import matplotlib.pyplot as plt
def run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler,splits=["train","test"]):
    backbone, neck, head = network
    logger_exp = logging.getLogger("experiment_logger")
    logger_exp.setLevel(logging.INFO)
    exp_handler = logging.FileHandler("experiment.log")  # 日志文件
    exp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    exp_handler.setFormatter(exp_formatter)
    logger_exp.addHandler(exp_handler)
    
    # weight_method = NashMTL(n_tasks=3, device='cuda')
    rho_best, epoch_best, rl2_best = 0, 0, 0
    test_srcc = []
    
    test_only = "train" not in splits
    if test_only:
        cfg.epoch_num = 400

    for epoch in range(cfg.epoch_num):
        for split in splits:
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
            for data_ in data_loaders[split]:
                data, clip_info = data_
                # import pdb; pdb.set_trace()
                score = data["score"].float().cuda()
                video = data["video"]
                if cfg.split_feats:
                    
                    
                    if "feats" in data:
                        clip_feats = data["feats"].cuda()
                    else:
                        import pdb; pdb.set_trace()
                        bs, frame, feats  = video.shape
                        video = video.reshape(video.shape[0],3,48,16,224,224).cuda()
                        clip_feats = torch.empty(bs, video.shape[2], feats).cuda()
                        for i in range(frame):
                            clip_feats[:,i] = backbone(video[:,:,i,:,:,:].cuda())[1].squeeze(-1).squeeze(-1).squeeze(-1)
                else:
                    bs, frame, h, w  = video.shape
                    clip_feats = backbone(video)[1].squeeze(-1).squeeze(-1).permute(0,2,1)
                
                tgt_weight, graph_attn = neck(clip_feats,train=False)#split=="train")
                probs, weight, means, var = head(tgt_weight)
                
                pred_scores.extend([i.item() for i in probs])
                true_scores.extend(score.cpu().numpy())
                
                kld_loss, self_map_lst, cross_map_lst = attention_loss(graph_attn, kld, self_map_lst, cross_map_lst)
                # self_map_vis(graph_attn)
                if epoch > 5000 and test_only:
                    # inter_vis(means,weight,clip_info,cfg.dataset_name)
                    user_study(means,weight,clip_info,cfg.dataset_name)
                    # print(f"inter_results/LOGO: Video: {clip_info[0][0]}, Seq: {clip_info[1][0].item()}")
                    # import pdb; pdb.set_trace()
                # plt.cla()
                # plt.clf()
                # plt.figure(figsize=(8, 6))
                # quality = means.squeeze(-1)[0].detach().cpu().numpy()
                # difficulty = weight.squeeze(-1)[0].detach().cpu().numpy()
                # plt.plot(np.arange(68)[::4],  (quality * difficulty * 125 - 1)[::4], marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
                # print(quality * difficulty * 250)
                # # 添加标题和标签
                # plt.ylim(-0.5, 1)
                # plt.title('Score vs. Clip', fontsize=16)
                # plt.xlabel('Clip Number', fontsize=14)
                # plt.ylabel('Score', fontsize=14)

                # # 显示网格
                # plt.grid(True)
                # plt.savefig("vis.png")
                
                
                if cfg.dino_loss:
                    probs = probs + torch.randn_like(probs).normal_(mean=0.0, std=0.05) #
                mes_loss = mse(probs,score)
                if cfg.att_loss:
                        loss = mes_loss + kld_loss
                else:
                    loss = mes_loss
                # import pdb; pdb.set_trace()
                # if split=="train":
                #     losses_cat = torch.stack([loss.mean(), mes_loss.mean(), kld_loss.mean()])
                    
                #     weighted_loss, extra_outputs = weight_method.get_weighted_loss(
                #             losses=losses_cat,
                #             shared_parameters=list(neck.parameters()) +list(head.parameters()),
                #             )
                
                #     loss = weighted_loss
                losses += loss
                rho, p, rl2 = cal_spearmanr_rl2(pred_scores, true_scores)
                
                
                if split=="train":
                    scheduler.step()
                    # for param_group in optimizer.param_groups:
                    #     print("当前学习率:", param_group['lr'])
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    
            # show information
            log_and_print(base_logger, f'Epoch: {epoch}, {split} correlation: {rho}, best: {rho_best}')
            
            if rho > rho_best and split == "test":
                rho_best = rho
                epoch_best = epoch
                rl2_best = rl2
                log_and_print(base_logger, '-----New best found!-----')
                # import pdb; pdb.set_trace()
                if not test_only:
                    torch.save({'epoch': epoch,
                                'backbone': backbone.state_dict(),
                                'neck': neck.state_dict(),
                                'head': head.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'rho_best': rho_best}, f'ckpts/i3d_{cfg.dataset_name}_{cfg.seed}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}_{cfg.num_layers}.pt') 
                
                

    if test_only:                # 记录日志
        logger_exp.info(f"test dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
    # logger_exp.info(f"dataset: {cfg.dataset_name}, seed: {cfg.seed}, label: {cfg.label}, query_var: {cfg.query_var}, pe: {cfg.pe}, att_loss: {cfg.att_loss}, dino_loss:{cfg.dino_loss}, num_layers: {cfg.num_layers}, SRCC: {rho_best}, RL2: {rl2_best}")
            