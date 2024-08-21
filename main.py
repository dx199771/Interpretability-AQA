import os
import torch.nn as nn
from utils.utils import parse_args, init_seed, init_gpu
from dataloader import build_dataloader
from models import build_backbone, build_neck, build_head

def main(cfg):
    
    # necessary modules
    init_seed(cfg)
    init_gpu(cfg)
    
    # dataloader
    dataloader = build_dataloader(cfg)
    
    # network
    backbone = build_backbone(cfg)
    neck = build_neck(cfg)
    head = build_head(cfg)
    
    if cfg.multi_gpu:
        backbone = nn.DataParallel(backbone)
        head = nn.DataParallel(head)
        neck = nn.DataParallel(neck)
        
    if cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        
    # loss function
    
    # optimizer and scheduler
    
    #
    


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)