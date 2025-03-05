import os
# import wandb
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils.utils import parse_args, init_seed, init_gpu, get_logger
from dataloader import build_dataloader
from models import build_backbone, build_neck, build_head
from run import run
from torchinfo import summary
def main(cfg):
    
    # necessary modules
    init_seed(cfg)
    multi_gpu, device = init_gpu(cfg)
    
    # dataloader
    data_loaders = build_dataloader(cfg)
    
    # network
    backbone = build_backbone(cfg)
    neck = build_neck(cfg)
    head = build_head(cfg)
    
    if multi_gpu:
        backbone = nn.DataParallel(backbone)
        head = nn.DataParallel(head)
        neck = nn.DataParallel(neck)
        
    # summary(head, input_size=(48,68,1024))
    # summary(neck, input_size=(48,68,1024))
    # state_dict = torch.load("model_rgb.pth")
    # new_state_dict = {"module." + k: v for k, v in state_dict.items()}
    # backbone.load_state_dict(new_state_dict)
    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        # 
        
        neck.load_state_dict(torch.load(cfg.load_from)["neck"],strict=False)
        head.load_state_dict(torch.load(cfg.load_from)["evaluator"],strict=False)
    
    # a = "/mnt/welles/scratch/datasets/condor/backup/detr-aqa-hinge/ckpts/gym_5_Clubs_True_2.0_query_pe.pt"
    # a = "/mnt/welles/scratch/datasets/condor/backup/detr-aqa-hinge/ckpts/gym_5_Clubs_True_2.0_query_pe.pt"
    # a = "ckpts/gym_5_Clubs_True_2.0_query_pe.pt"
    # head.load_state_dict(torch.load(a)["evaluator"])
    
    # loss function
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='sum')
    
    # optimizer and scheduler
    optimizer = torch.optim.Adam([
        {'params': backbone.parameters(), 'lr': cfg.lr},
        {'params': neck.parameters(), 'lr': cfg.lr},
        {'params': head.parameters(), 'lr':cfg.lr}],
        lr=cfg.lr,
        weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
    
    # log and wandb
    
    base_logger = get_logger(f'exp/{cfg.seed}_{cfg.dataset_name}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}.log', "EXP1")    
    # wandb.init(project="aqa-detr-model",name=f"{cfg.seed}_{cfg.dataset_name}_{cfg.label}_{cfg.att_loss}_{cfg.query_var}_{cfg.pe}")
    # wandb.config = {"learning_rate": cfg.lr, "epochs": cfg.epoch_num, "batch_size": cfg.bs_train}
    
    network = [backbone, neck, head]
    if multi_gpu:
        network = [backbone, neck, head]
    run(cfg, base_logger, network, data_loaders, kld, mse, optimizer, scheduler)


if __name__ == '__main__':
    cfg = parse_args()
    

    print(f"PE: {cfg.pe}, Query Var: {cfg.query_var}, Att Loss: {cfg.att_loss}, Dino Loss: {cfg.dino_loss}")
    
    main(cfg)