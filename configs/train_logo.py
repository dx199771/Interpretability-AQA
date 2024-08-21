
seed = 1
multi_gpu = "0,1"

# dataset config (logo, gym, fisv)
dataset_name = "logo"
swim_dir = "/mnt/welles/scratch/datasets/condor/backup/logo/Video_result" #"./data/logo/Video_result"
swim_label = "/mnt/welles/scratch/datasets/condor/backup/logo/LOGO Anno&Split" #"./data/logo/LOGO Anno&Split"
presave = None #"./data/logo/LOGO Anno&Split"

q_number = 48

# dataloader config
subset = 0
bs_train = 1024
bs_test = 1024
num_workers = 4

# network config (i3d, vivit)
backbone = "i3d"

i3d = dict(
    backbone="I3D",
    neck="",
    evaluator="",
    I3D_ckpt_path="/home/xu/repo/CoRe/MTL-AQA/model_rgb.pth" 
)

vivit = dict(
    backbone="ViViT",
)

neck = "TQN"
head = "weighted"

# training config
load_from = None
