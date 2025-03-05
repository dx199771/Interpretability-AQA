import os
import torch
import random
import logging
import numpy as np
import argparse

def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger

def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)

def init_seed(args):
    """
    Set random seed for torch and numpy.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='training config parser')
    parser.add_argument(
        '--config',
        help='train config file path',
        default='configs/train_logo.py')
    parser.add_argument("--pe", type=str, default="default_pe")
    parser.add_argument("--query_var", type=float, default=1)
    parser.add_argument("--att_loss", action="store_true", default=False)
    parser.add_argument("--dino_loss", action="store_true", default=False)
    parser.add_argument("--label", type=str, default="TES")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load_from", type=str, default="")
    args = parser.parse_args()
    
    
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    cfg.pe = args.pe
    cfg.load_from = args.load_from
    cfg.query_var = args.query_var
    cfg.att_loss = args.att_loss
    cfg.dino_loss = args.dino_loss
    cfg.seed = args.seed
    if "label" in cfg:
        cfg.label = args.label
    else:
        cfg.label = "logo"
    return cfg

def init_gpu(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.multi_gpu
    multi_gpu = False if len(cfg.multi_gpu.split(",")) <= 0 else True
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return multi_gpu, device