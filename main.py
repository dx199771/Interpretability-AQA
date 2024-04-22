import os

import torch

from opts import *
from utils import *





def main():
    # configs
    args = parse_opt()
    init_seed(args)

    # gpu settings
    os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_gpu
    multi_gpu = False if len(args.multi_gpu.split(",")) <= 0 else True
    if multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders
    if args.dataset == "logo":
        from datasets.dataloader_logo import *
        data_loaders = get_dataloader(args)
    elif args.dataset == "gym":
        from datasets.dataloader_gymnastic import *
        data_loaders = get_dataloader(args)
    
    # networks
    backbone = I3D(num_classes = 400, modality = 'rgb')#.cuda()

if __name__ == '__main__':
    main()