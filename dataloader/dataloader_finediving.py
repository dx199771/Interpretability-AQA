import os
import torch
import glob
import pickle
import numpy as np
from PIL import Image
import torch.nn.functional as F
from utils.dataloader_utils import worker_init_fn, get_video_trans



class Finediving_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        self.subset = subset
        self.transforms = transform
        self.data_dir = args.finediving_dir

        self.read_data_label(args)
        if self.subset == "train":
            self.dataset = self.train_split    
        else:
            self.dataset = self.test_split

    def read_data_label(self,args):
        self.label_file = {}
        for game in glob.glob(os.path.join(self.data_dir, "*")):
            base_game = os.path.basename(game)
            for clip in glob.glob(os.path.join(game, "*")):
                base_clip = int(os.path.basename(clip))
                self.label_file[(base_game,base_clip)] = glob.glob(os.path.join(clip,"*.jpg"))
        with open(os.path.join(args.finediving_label,"fine-grained_annotation_aqa.pkl"), "rb") as f:
            self.aqa_anno_dict = pickle.load(f)
        with open(os.path.join(args.finediving_label,"FineDiving_fine-grained_annotation.pkl"), "rb") as f:
            self.anno_dict = pickle.load(f)
        with open(os.path.join(args.finediving_label,"train_split.pkl"), "rb") as f:
            self.train_split = pickle.load(f)
        with open(os.path.join(args.finediving_label,"test_split.pkl"), "rb") as f:
            self.test_split = pickle.load(f)

    def load_img_seq(self, clip_name):

        image_list = sorted(self.label_file[clip_name])    
        frame_list = np.linspace(0, len(image_list)-1, 96).astype(np.int32)
        crop_frame = [image_list[i] for i in frame_list]
        video = [Image.open(image_path) for image_path in crop_frame]
    
        #def get_action_label(self,clip):
        crop_label = [self.aqa_anno_dict[clip_name][4][i] for i in frame_list]
        action_labels = crop_label#self.aqa_anno_dict[clip_name][4]        
        return self.transforms(video), action_labels
        
    def __getitem__(self,index):
        data = {}
        clip = self.dataset[index]
        data["video"], data["actions"]= self.load_img_seq(clip)
        data["actions"] = F.one_hot(torch.tensor(data["actions"]).long()-1,37)
        data["score"] = self.anno_dict[clip]["dive_score"] #/ 114.8
        
        return data, clip
    def __len__(self,):
        return len(self.dataset)
    
def get_dataloader(args):   
    data_loaders = {}
    train_trans, test_trans = get_video_trans()
    data_loader_train = Finediving_Dataset(args,"train",train_trans)
    data_loader_test = Finediving_Dataset(args,"test",test_trans)
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


