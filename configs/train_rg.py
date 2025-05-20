
seed = 5
multi_gpu = "0"

# dataset config (logo, gym, fisv)
dataset_name = "gym"

gymnastic_dir = "/mnt/welles/scratch/datasets/condor/backup/detr-aqa/GDLT_data"
gymnastic_data = "/mnt/welles/scratch/xu/gymnastics_imgs_resize/images"
# if save backbone feature load here
presave_feature = "/mnt/welles/scratch/datasets/condor/backup/detr-aqa/GDLT_data/swintx_avg_fps25_clip32"
# dataloader config
subset = 0
bs_train = 48
bs_test = 1
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
q_number = 68
# variange for initilize query
query_var = 2
# positional embedding method ["query_pe","query_memory_pe","no"]
pe = "query_pe" 
att_loss = True
dino_loss = True
max_len = 103

label = "Ribbon"
num_layers = 2

# log and wandb

# training config
split_feats = True
epoch_num = 1000
# load_from = "/mnt/fast/nobackup/scratch4weeks/xd00101/detr-aqa-hinge/ckpts/logo_i3d_05drop_decoder_2layers_decoupled_20query_softmaxweight_augment_100score_tpti3d_warmup_onlyscore_seed199771_uncertainty_6layers1.pt"
lr = 1e-4
