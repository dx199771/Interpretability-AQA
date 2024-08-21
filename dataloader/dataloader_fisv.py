import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from utils.dataloader_utils import worker_init_fn, get_video_trans

class Fisv_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        self.subset = subset
        self.transforms = transform
        
        self.label_target = args.fisv_label
        self.data =  args.fisv_dir
        self.read_data_label(args)
        if self.subset == "train":
            self.dataset = self.train_file   
            self.label = self.train_label 
        else:
            self.dataset = self.test_file
            self.label = self.test_label
        self.image_list_test = []
        self.presave = True
    def read_data_label(self,args):
    
        self.train_file = []
        self.test_file = []
        self.train_label = {}
        self.test_label = {}
        with open(os.path.join(args.fisv_dir,"train.txt"),"r") as f:
            train_label_raw = f.readlines()
        for i in train_label_raw:            
            if self.label_target == "TES":        
                clip_score = float(i.split("\t")[0].split(" ")[1])
            elif self.label_target == "PCS":        
                clip_score = float(i.split("\t")[0].split(" ")[2])
            video_name = i.split(" ")[0]
            self.train_label[video_name] = clip_score

        with open(os.path.join(args.fisv_dir,"test.txt"),"r") as f:
            test_label_raw = f.readlines()
        for i in test_label_raw:
            if self.label_target == "TES":
                clip_score = float(i.split("\t")[0].split(" ")[1])
            elif self.label_target == "PCS":        
                clip_score = float(i.split("\t")[0].split(" ")[2]) 
            video_name = i.split(" ")[0]
            self.test_label[video_name] = clip_score

    def load_img_seq(self, image_list):
       
        self.image_list_test.append(len(image_list))
        new_img_list = np.linspace(0, len(image_list) - 1, 768)
        
        new_img_list = [image_list[int(i)] for i in new_img_list]
        video = [Image.open(image_path) for image_path in new_img_list]
        
        return self.transforms(video)
    def __getitem__(self, index):  
        data = {}
        if not self.presave:
            clip = self.dataset[index][0]
            clip_file = self.dataset[index][1]
        else:
            clip = list(self.label)[index]
        
        
        data["feats"] = np.load(os.path.join(self.data,"swintx_avg_fps25_clip32",f"{clip}.npy"))

        if len(data["feats"]) > 136:
            st = np.random.randint(0, len(data["feats"]) - 136)
            data["feats"] = data["feats"][st:st + 136]
        elif len(data["feats"]) < 136:
            new_feat = np.zeros((136, data["feats"].shape[1]))
            new_feat[:data["feats"].shape[0]] = data["feats"]
            data["feats"] = new_feat

        data["video"] = torch.tensor(0)
        data["score"] = self.label[clip] / 43
        return data, clip
    
    def __len__(self,):
        return len(self.label)
        
def get_dataloader(args):   
    data_loaders = {}
    train_trans, test_trans = get_video_trans()
    data_loader_train = Fisv_Dataset(args,"train",train_trans)
    data_loader_test = Fisv_Dataset(args,"test",test_trans)
    if args.subset > 1:
        subset = list(range(0, len(data_loader_train), args.subset))
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
 