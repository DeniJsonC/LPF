""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
from tkinter import X
import torch
import torch.nn as nn
from thop import profile

from model.utils import LayerNorm,Mlp

import torch.nn.functional as F

class SAWF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=4,bias=False):
        super(SAWF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)       
        self.sff = Mlp(in_channels, d, self.height)
        self.sca=Mlp(in_channels,d,in_channels*self.height)
    def forward(self, fusion_vectors):
        B, C, H, W = fusion_vectors.shape
        sff = self.sff(fusion_vectors).softmax(dim=1).permute(1,0,2,3).unsqueeze(-3)# height*B*1*H*W

        sca=F.adaptive_avg_pool2d(fusion_vectors,output_size=1)
        sca=self.sca(sca).reshape(B, C, self.height).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)#self.height*B*C*1

        attn=[]
        for i in range(self.height):
            attn.append(sff[i]*sca[i])
        del sff,sca,fusion_vectors
        return attn

class WEB(nn.Module):
    def __init__(self, in_dim,out_dim,height=5,bias=False,proj_drop=0.,norm=LayerNorm):
        super().__init__()
        
        self.norm=norm(in_dim)
        self.fcs = nn.ModuleList([])
        self.height=height
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1,padding=0,bias=bias))

        self.sin_f=nn.Conv2d(out_dim,out_dim,kernel_size=(7,7),stride=1,padding=(7//2,7//2),groups=out_dim,bias=bias)
        self.cos_f=nn.Conv2d(out_dim,out_dim,kernel_size=(7,7),stride=1,padding=(7//2,7//2),groups=out_dim,bias=bias)

        self.saff=SAWF(out_dim,height=3)

        self.proj = nn.Conv2d(out_dim, out_dim, 1, 1,bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
 
    def forward(self, x):
        x=self.norm(x)
        ex_ch=[fc(x) for fc in self.fcs]

        cos_f=ex_ch[0]*torch.cos(ex_ch[1])
        sin_f=ex_ch[2]*torch.sin(ex_ch[3])

        cos_f=self.cos_f(cos_f)
        sin_f=self.sin_f(sin_f)

        u=cos_f+sin_f+ex_ch[4]
        attn=self.saff(u)
        del u
        x=cos_f*attn[0]+sin_f*attn[1]+ex_ch[4]*attn[2]
      
        del cos_f,sin_f,attn

        del ex_ch

        x = self.proj(x)
        x = self.proj_drop(x)           
        return x

class FEM(nn.Module):

    def __init__(self, norm_dim,dim,height=6, mlp_ratio=2., bias=False,  attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=LayerNorm,):
        super().__init__()
        self.conv=nn.Conv2d(norm_dim,dim,1,1,0,bias=False)
        self.fdeb = WEB(norm_dim,dim, height=height,bias=bias,norm=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = self.conv(x)+ self.drop_path(self.fdeb(x)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

def basic_blocks(embed_dims, index, layers, height=6,mlp_ratio=3., bias=False,  attn_drop=0.,
                 drop_path_rate=0.,norm_layer=LayerNorm, **kwargs):
    blocks = []
    norm_index=0
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index!=0:
            norm_index=index-1
            blocks.append(FEM(embed_dims[norm_index],embed_dims[index],height, mlp_ratio=mlp_ratio, bias=bias, 
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer))
    blocks = nn.Sequential(*blocks)
    return blocks 

class LPF(nn.Module):
    def __init__(self,in_channel=3,embed_dims=[32,64,48,32],layers=[1,1,1,1],mlp_ratios=[1,0.5,0.5,1], 
        bias=False, attn_drop_rate=0., drop_path_rate=0.,height=5,task=None,
        norm_layer=LayerNorm):
        super().__init__()
        network=[]
        self.conv=nn.Conv2d(in_channel,embed_dims[0],kernel_size=3,stride=1,padding=1,bias=False)
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims, i,layers, height,mlp_ratio=mlp_ratios[i], bias=bias,
                                 attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            network.append(stage)
            if i >= len(layers) - 1:
                break
        self.network = nn.ModuleList(network)
        self.head=nn.Conv2d(embed_dims[-1],3,3,1,1,bias=False)
    def forward(self,x):
        raw_x=x
        x=self.conv(x)
        for idx,block in enumerate(self.network):
            x=block(x)
        x = self.head(x)
        return x+raw_x
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=LPF()
    model = model.to(device)
    if str(device) =='cuda':
        input=torch.randn(1,3,256,256).cuda()
    else:
        input=torch.randn(1,3,256,256)
    print(model)
    flops,params=profile(model,inputs=(input,))
    print('flops:{}G params:{}M'.format(flops/1e9,params/1e6))