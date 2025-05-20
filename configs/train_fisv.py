
seed = 999
multi_gpu = "1"

# dataset config (logo, gym, fisv)
dataset_name = "fisv"
# swim_dir = "/mnt/welles/scratch/datasets/condor/backup/logo/Video_result" #"./data/logo/Video_result"
# swim_label = "/mnt/welles/scratch/datasets/condor/backup/logo/LOGO Anno&Split" #"./data/logo/LOGO Anno&Split"
# presave =  "/mnt/welles/scratch/datasets/condor/backup/logo/logo_feats"


fisv_dir = "/mnt/welles/scratch/datasets/condor/data/fis-v"
label = "TES" #PCS

# dataloader config
subset = 0
bs_train = 32
bs_test = 32
num_workers = 4

# network config (i3d, vivit)
backbone = "vivit"

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
# query number
q_number = 136
# variange for initilize query
query_var = 5
# positional embedding method ["query_pe","query_memory_pe","no"]
pe = "query_pe" 
att_loss = True
dino_loss = True
max_len = 136

num_layers = 2



# training config
split_feats = True
epoch_num = 1000
# load_from = "/mnt/fast/nobackup/scratch4weeks/xd00101/detr-aqa-hinge/ckpts/logo_i3d_05drop_decoder_2layers_decoupled_20query_softmaxweight_augment_100score_tpti3d_warmup_onlyscore_seed199771_uncertainty_6layers1.pt"
lr = 1e-4
