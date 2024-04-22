import os
import pdb
import glob
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

class Gym_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        self.subset = subset
        self.transforms = transform
        
        self.label_target = args.gym_label
        self.read_data_label(args)

        if self.subset == "train":
            self.dataset = self.train_file   
            self.label = self.train_label 
        else:
            self.dataset = self.test_file
            self.label = self.test_label
        self.presave = args.gym_presave
        self.image_list_test = []

    def read_data_label(self,args):
        self.train_file = []
        self.test_file = []
        self.train_label = {}
        self.test_label = {}
        with open(os.path.join(args.gymnastic_dir,"train.txt"),"r") as f:
            train_label_raw = f.readlines()
        for i in train_label_raw[1:]:
            if i.split("\t")[0].split("_")[0] != self.label_target:
                continue
            video_name = i.split("\t")[0]
            self.train_label[video_name] = float(i.split("\t")[3])
            self.train_file.append([video_name,sorted(glob.glob(os.path.join(args.gymnastic_data,video_name,"*")))])

        with open(os.path.join(args.gymnastic_dir,"test.txt"),"r") as f:
            test_label_raw = f.readlines()
        for i in test_label_raw[1:]:
            if i.split("\t")[0].split("_")[0] != self.label_target:
                continue
            video_name = i.split("\t")[0]
            self.test_label[video_name] = float(i.split("\t")[3])
            self.test_file.append([video_name,sorted(glob.glob(os.path.join(args.gymnastic_data,video_name,"*")))])

    def load_img_seq(self, image_list):
        if self.presave:
            feats = np.load(os.path.join(self.presave,f"{clip}.npy"))
            if len(feats) > 68:
                st = np.random.randint(0, len(feats) - 68)
                feats = feats[st:st + 68]
            elif len(feats) < 68:
                new_feat = np.zeros((68, feats.shape[1]))
                new_feat[:feats.shape[0]] = feats
                feats = new_feat
            return feats
        
        self.image_list_test.append(len(image_list))
        new_img_list = np.linspace(0, len(image_list) - 1, 768)
        
        new_img_list = [image_list[int(i)] for i in new_img_list]
        video = [Image.open(image_path) for image_path in new_img_list]
        
        return self.transforms(video)
    
    def __getitem__(self, index):  
        data = {}
        clip = self.dataset[index][0]
        clip_file = self.dataset[index][1]       
        data["video"] = self.load_img_seq(clip_file)        
        data["score"] = self.label[clip] / 25
        
        return data, clip
    
    def __len__(self,):
        return len(self.dataset)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataloader(args):   
    data_loaders = {}
    train_trans, test_trans = get_video_trans()
    data_loader_train = Gym_Dataset(args,"train",train_trans)
    data_loader_test = Gym_Dataset(args,"test",test_trans)
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
