import os
import torch
import random
import logging
import numpy as np
import argparse
from mmcv import Config
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
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)
   
    return cfg

def init_gpu(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.multi_gpu
    multi_gpu = False if len(cfg.multi_gpu.split(",")) <= 0 else True
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return multi_gpu, device