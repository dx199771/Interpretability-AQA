"""
Code modified from DETR tranformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""

import copy
from typing import Optional, List
import pickle as cp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .deformatt import DeformAttn
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,global_feat=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.global_feat = global_feat
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                train=True):
        output = tgt
        T,B,C = memory.shape
        intermediate = []
        outputs = []
        self_maps = []
        cross_maps = []
        memorys = []
        if self.global_feat:
            memory = torch.cat((memory, memory.mean(0).unsqueeze(0)))
        for n,layer in enumerate(self.layers):
            
            residual=True
            if n ==1 and train==False:
                plot =True
            else:
                plot= False
            
            
            output,ws,self_map,cross_map, memory = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,residual=residual, plot=plot)
            outputs.append(output)
            self_maps.append(self_map)
            cross_maps.append(cross_map)
            memorys.append(memory)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, self_maps, cross_maps, memorys



class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.5,
                 activation="relu", normalize_before=False):
        super().__init__()
        #import pdb; pdb.set_trace()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # self.conv1d = nn.Conv1d(in_channels=48, out_channels=48, kernel_size=3, padding=1)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     residual=True):
        
        q = k = self.with_pos_embed(tgt, query_pos)
        
        
        tgt2,ws = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        
        
        tgt = self.norm1(tgt)
        tgt2,ws = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        

        # attn_weights [B,NUM_Q,T]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt,ws

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    plot = False):
        tgt2 = self.norm1(tgt)

        
        # plt.hist( tgt2[:,0,:].detach().cpu().numpy(),alpha=0.5)
        # plt.savefig("hist.png")


        #plt.hist(tgt2[0,0].detach().cpu().numpy())
        #plt.savefig("hist.png")
        #
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        
        # q = k =tg torch.cat((q, q.mean(0).unsqueeze(0)))
        self_map,ws = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        # self_map = self.conv1d(self_map.permute(1,0,2)).permute(1,0,2)
        
        # import pdb; pdb.set_trace()  
        #import pdb; pdb.set_trace()
        if False:#plot:
            aa = F.softmax(torch.div(torch.matmul(self_map.transpose(0,1), self_map.transpose(0,1).transpose(-1,-2)), self_map.size(-1)),dim=1).cpu().detach().numpy()
            #aa = torch.sqrt(torch.matmul(self_map.transpose(0,1), self_map.transpose(0,1).transpose(-1,-2)))
            # import pdb; pdb.set_trace()
            
            plt.cla()
            plt.clf()
            plt.imshow(aa[0], cmap='hot', interpolation='nearest')
            
            
        # #     #print(aa)
            plt.colorbar()
            plt.savefig("atmap.png")
            # plt.close()
        #     #
        tgt = self_map + self.dropout1(self_map)
        
        tgt2 = self.norm2(tgt)
        
        
        cross_map,attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos), #.transpose(0,1)
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        #import pdb; pdb.set_trace()
        qq = cross_map
        kk = self.with_pos_embed(memory, pos)
        # if plot:
        #     plt.cla()
        #     plt.clf()
        #     #import pdb; pdb.set_trace()
        
        #     aa = F.softmax(torch.div(torch.matmul(cross_map.transpose(0,1), cross_map.transpose(0,1).transpose(-1,-2)), cross_map.size(-1)),dim=1)
        #     #aa = torch.sqrt(torch.matmul(cross_map.transpose(0,1), memory.transpose(0,1).transpose(-1,-2)))
        #     plt.imshow(aa.cpu().detach().numpy()[0], cmap='hot', interpolation='nearest')
        #     plt.colorbar()
        #     plt.savefig("atmap2.png")
        tgt = tgt + self.dropout2(cross_map)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt,ws, self_map, qq, kk

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                residual=True,
                plot=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,plot)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,residual, plot)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")