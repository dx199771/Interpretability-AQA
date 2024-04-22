import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--multi_gpu', type=str, default="0") #"0,1"

    #dataloader 
    parser.add_argument('--dataset', type=str, default="logo") # finediving, logo, gym
    parser.add_argument('--subset', type=int, default=0)
    parser.add_argument('--bs_train', type=int, default=2)#6
    parser.add_argument('--bs_test', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)#12

    #logo dataloader
    parser.add_argument('--swim_dir', type=str, default="/mnt/welles/scratch/datasets/LOGO/Video_result")
    parser.add_argument('--swim_label', type=str, default="/mnt/fast/nobackup/scratch4weeks/xd00101/logo/LOGO Anno&Split")
    parser.add_argument('--logo_presave', type=str, default="/mnt/fast/nobackup/scratch4weeks/xd00101/logo/logo_feats/")

    #gymnastic dataloader
    parser.add_argument('--gymnastic_dir', type=str, default="/mnt/fast/nobackup/scratch4weeks/xd00101/detr-aqa/GDLT_data")#12
    parser.add_argument('--gymnastic_data', type=str, default="/mnt/welles/scratch/xu/gymnastics_imgs_resize/images")#12
    parser.add_argument('--gym_label', type=str, default="Clubs") #Ball #Clubs #Hoop #Ribbon
    parser.add_argument('--gym_presave', type=str, default="/mnt/fast/nobackup/scratch4weeks/xd00101/logo/logo_feats/")


    
    args = parser.parse_args()

    return args
