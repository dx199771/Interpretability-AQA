import os
import pdb
import glob
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvideotransforms import video_transforms, volume_transforms

def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((224, 224)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((224, 224)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans

class Logo_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        self.subset = subset
        self.transforms = transform

        self.data_dir = args.swim_dir
        self.read_data_label(args)

        if self.subset == "train":
            self.dataset = self.train_split    
        else:
            self.dataset = self.test_split
        self.presave = args.logo_presave

    def load_img_seq(self, clip_name):
        if self.presave:
            feats = np.load(f"{self.presave}/{clip_name[0]}_{clip_name[1]}.npy",allow_pickle=True)
            return feats, None
        else:
            image_list = sorted(self.label_file[clip_name])
            selected_indices = np.linspace(0, len(image_list) - 1, 768, dtype=int)
            selected_frames = [image_list[i] for i in selected_indices]
            video = [Image.open(image_path) for image_path in selected_frames]

            action_labels = self.anno_dict[clip_name][4]
            selected_labels = [action_labels[i] for i in selected_indices]
            
            return self.transforms(video), selected_labels

    def read_data_label(self,args):
        self.label_file = {}
        for game in glob.glob(os.path.join(self.data_dir, "*")):
            base_game = os.path.basename(game)
            for clip in glob.glob(os.path.join(game, "*")):
                base_clip = int(os.path.basename(clip))
                self.label_file[(base_game,base_clip)] = glob.glob(os.path.join(clip,"*.jpg"))

        with open(os.path.join(args.swim_label,"anno_dict.pkl"), "rb") as f:
            self.anno_dict = pickle.load(f)
        
        with open(os.path.join(args.swim_label,"formation_dict.pkl"), "rb") as f:
            self.formation_dict = pickle.load(f)
        
        with open(os.path.join(args.swim_label,"train_split3.pkl"), "rb") as f:
            self.train_split = pickle.load(f)

        with open(os.path.join(args.swim_label,"test_split3.pkl"), "rb") as f:
            self.test_split = pickle.load(f)
    
    
    def __getitem__(self, index):  
        data = {}
        clip = self.dataset[index]
        data["video"], data["actions"] = self.load_img_seq(clip) #self.load_img_seq(clip).reshape(3,128,16,224,224)
        data["actions"] = torch.tensor(1) #F.one_hot(torch.tensor(data["actions"]).long()-1,12)
        data["score"] = self.anno_dict[clip][1] / 100

        return data, clip
    
    def __len__(self,):
        return len(self.dataset)
    
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataloader(args):   
    data_loaders = {}
    train_trans, test_trans = get_video_trans()
    data_loader_train = Logo_Dataset(args,"train",train_trans)
    data_loader_test = Logo_Dataset(args,"test",test_trans)
    if args.subset > 1:
        subset = list(range(0, len(data_loader_train), args.subset))
        #odds = list(range(1, len(trainset), 2))
        data_loader_train = torch.utils.data.Subset(data_loader_train, subset)
    
    train_dataloader = torch.utils.data.DataLoader(data_loader_train, batch_size=args.bs_train,
                                            shuffle=True,num_workers = int(args.num_workers),
                                            pin_memory=True, worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(data_loader_test, batch_size=args.bs_test,
                                            shuffle=False,num_workers = int(args.num_workers),
                                            pin_memory=True)
    data_loaders["train"] = train_dataloader
    data_loaders["test"] = test_dataloader

    return data_loaders

if __name__ == "__main__":
    from opts import parse_opt
    args = parse_opt()
    dataloader = get_dataloader(args)["train"]
    for i in dataloader:
        pdb.set_trace()
