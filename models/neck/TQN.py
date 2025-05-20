import torch
import pdb
from .transformer import *
import torch.nn as nn
import math
from torch.autograd import Variable

import torch.nn.init as init

def tfm_mask(seg_per_video,temporal_mutliplier=1):
    """
    Attention mask for padded sequence in the Transformer
    True: not allowed to attend to 
    """
    B = len(seg_per_video)
    L = 1024 #max(seg_per_video) * temporal_mutliplier
    mask = torch.ones(B,L,dtype=torch.bool)
    for ind,l in enumerate(seg_per_video):
        mask[ind,:(l*temporal_mutliplier)] = False

    return mask


class TQN(nn.Module):
    def __init__(self,d_model, num_queries, query_var=1, pe="no", H=4, N=2, feature_dim=1024, max_len=136):
        super(TQN, self).__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.query_var = query_var
        self.pe = pe
        self.max_len = max_len
        #encoder_layer = TransformerEncoderLayer(self.d_model, H, 1024,
                                                #0.5, 'relu', normalize_before=True)
        #encoder_norm = nn.LayerNorm(d_model)
        #self.encoder = TransformerEncoder(encoder_layer, 4, encoder_norm)
        
        global_feat = False
        decoder_layer = TransformerDecoderLayer(self.d_model, H, 1024,
                                        0.7, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, N, decoder_norm,
                                  return_intermediate=False,global_feat=global_feat)
        if global_feat == True:
            self.query_embed = nn.Embedding(self.num_queries+1,self.d_model)
        else:
            self.query_embed = nn.Embedding(self.num_queries,self.d_model)
        #query_embed = nn.Embedding(num_embeddings=6*6, embedding_dim=1024).cuda()
        #input_indices = torch.LongTensor([[i*6 + j for j in range(6)] for i in range(6)]).cuda()
        #self.query_embed = query_embed(input_indices)

        self.dropout_feas = nn.Dropout(0.5)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len= self.max_len)
       
        # self.query_mu = nn.Parameter(torch.zeros(self.num_queries, self.d_model))
        # self.query_logvar = nn.Parameter(torch.full((self.num_queries, self.d_model), -3.0))
        # self.query_fc = nn.Linear(1024,1024)
    def forward(self, input, train=False):
        input = input.float()
        B = len(input)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1,B,1)#.permute(1,0,2)# self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
       
        if self.query_var != 1:
            if False:
                query_mu = self.query_mu.unsqueeze(1).repeat(1,B,1)
                query_logvar = self.query_logvar.unsqueeze(1).repeat(1,B,1)
                std = torch.exp(0.5 * query_logvar)
                noise = torch.randn_like(query_mu).cuda()
                query_embed = query_mu + std * noise
                
            
            query_embed = init.normal_(query_embed, mean=0, std=self.query_var)
            # query_embed = self.query_fc(query_embed)
        # import pdb; pdb.set_trace()
        if self.pe == "query_pe":
            pe = None#self.pos_encoder(input[0,:,:]).repeat(B,1,1).transpose(0,1) #+- torch.min(self.pos_encoder(input[0,:,:]).repeat(B,1,1).transpose(0,1))
            query_pe = self.pos_encoder(query_embed[:,0,:]).repeat(B,1,1).transpose(0,1) #+- torch.min(self.pos_encoder(query_embed[:,0,:]).repeat(B,1,1).transpose(0,1))
        elif self.pe == "query_memory_pe":
            pe = self.pos_encoder(input[0,:,:]).repeat(B,1,1).transpose(0,1) #+- torch.min(self.pos_encoder(input[0,:,:]).repeat(B,1,1).transpose(0,1))
            query_pe = self.pos_encoder(query_embed[:,0,:]).repeat(B,1,1).transpose(0,1)
        elif self.pe == "no":
            pe = None
            query_pe = None
        elif self.pe == "memory_pe":
            pe = self.pos_encoder(input[0,:,:]).repeat(B,1,1).transpose(0,1)
            query_pe = None
        
        input_ = input.transpose(0,1)
        
        # query_embed = query_embed + query_pe
        features, self_maps, cross_maps, memorys = self.decoder(query_embed, input_, 
                memory_key_padding_mask=None, pos=pe, query_pos=query_pe,train=train)
                
        # features = self.dropout_feas(features)
        return features.permute(1,0,2), (self_maps, cross_maps, memorys)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=103):#136
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        pe = Variable(self.pe[:, :x.size(0)],
                      requires_grad=False)
        
        x = x + pe
        return self.dropout(x)
