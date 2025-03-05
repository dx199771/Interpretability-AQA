import torch
import numpy as np
import os
import pickle
import glob
from PIL import Image
from scipy import stats
import torch.nn.functional as F
from utils.dataloader_utils import worker_init_fn, get_video_trans


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
        
        self.presave = True
        self.presave_path = args.presave
        # import pdb; pdb.set_trace
        # if self.presave:
        #     self.presave_data = np.load(f"./{self.subset}.npy",allow_pickle=True)
        
    def load_img_seq(self, clip_name):
        

        if self.presave:
            video = np.load(f"{self.presave_path}/{clip_name[0]}_{clip_name[1]}.npy",allow_pickle=True)
            return video, None
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
        # print(1)
        data = {}
        clip = self.dataset[index]
        # import pdb; pdb.set_trace()
        data["video"], data["actions"] = self.load_img_seq(clip) 
        data["actions"] = torch.tensor(1) #dummy
        data["score"] = self.anno_dict[clip][1] / 100
        if self.presave:
            data["feats"] = data["video"]
        return data, clip
    def __len__(self,):
        return len(self.dataset)

def get_dataloader(args):   
    data_loaders = {}
    train_trans, test_trans = get_video_trans()
    data_loader_train = Logo_Dataset(args,"train",train_trans)
    data_loader_test = Logo_Dataset(args,"test",test_trans)
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
