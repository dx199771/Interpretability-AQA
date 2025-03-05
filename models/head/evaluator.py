import torch.nn as nn
import torch

class Evaluator_weighted(nn.Module):
    def __init__(self,):
        super(Evaluator_weighted, self).__init__()
        self.layer1_mean = nn.Linear(1024, 512)
        self.layer2_mean = nn.Linear(512, 256)
        self.layer3_mean = nn.Linear(256, 1)
        self.layer1_weight = nn.Linear(1024, 512)
        self.layer2_weight = nn.Linear(512, 256)
        self.layer3_weight = nn.Linear(256, 1)
    
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-2)
        
    def forward(self, x):
        x_mean = torch.relu(self.layer1_mean(x))
        x_mean = torch.relu(self.layer2_mean(x_mean))
        x_mean = self.layer3_mean(x_mean)

        x_weight = torch.relu(self.layer1_weight(x))
        x_weight= torch.relu(self.layer2_weight(x_weight))
        x_weight = self.softmax(self.layer3_weight(x_weight))
            
        logits = torch.sum(x_mean.squeeze(-1) * x_weight.squeeze(-1),axis=1)
        
        return logits, x_weight, x_mean, None
