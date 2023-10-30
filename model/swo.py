import torch
import torch.nn as nn
from wavemix import Level1Waveblock
from .hat_base import Upsample
from .hat_arch import OCAB
from .tool import calculate_rpi_oca
from .atn.sma import SMA

class CBLK(nn.Module):
    def __init__(self, inChannels, outChannels, k = 3, s = 1, p = 1, bias = True) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size = k, stride=s, padding = p,  bias = bias),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
class GNSiLU(nn.Module):
    def __init__(self, out_channels, groupHead = 8, nonlinearity = True, init_zero = False):
        super(GNSiLU, self).__init__()
        self.norm  = nn.GroupNorm(groupHead, out_channels)
        if nonlinearity:
            self.act  =  nn.SiLU()
        else:
            self.act  =  None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out  =  self.norm(input)
        if self.act is not None:
            out  =  self.act(out)
        return out
class Decode(nn.Module):
    def __init__(self, inChannels, outChannels, mult = 1, dropout = 0.3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Sequential(
            Level1Waveblock(mult = mult, ff_channel = inChannels, final_dim = outChannels, dropout = dropout),
        )
        
    def forward(self, x, y):
        x = torch.cat([self.up(x), y],1)
        x = x + self.conv(x)
        return x

class SKNConv(nn.Module):            
    def __init__(self, features_list, out_features, G = 8, r = 2, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            G: num of convolution groups.                                8
            r: the radio for compute d, the length of z.                 2      
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32
        """
        super(SKNConv, self).__init__()
        features = out_features
        d = max(int(features/r), L)
        self.M = len(features_list)
        self.features = features
        self.convs = nn.ModuleList()
        for i in range(self.M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features_list[i], out_features, kernel_size=3, stride=1, padding=1, groups=G),
                nn.BatchNorm2d(out_features),
                # nn.ReLU()
            ))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList()
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x[i]).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
          
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze( dim=-1).squeeze(dim=-1)
       
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
           
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class ATOCA(nn.Module):
    def __init__(self, inChannel, ouChannel, r = 1, w = 4, h = 4) -> None:
        super().__init__()
        middleChannel = inChannel//r
        """
        super parameters:
        ref :  git clone https://github.com/XPixelGroup/HAT.git 
        paper ref: https://arxiv.org/pdf/2309.05239.pdf
        """
        numHead = 6
        self.w = w
        self.h = h
        self.rpiTable = calculate_rpi_oca( window_size = self.w)
        self.ocatn = OCAB(
                dim = middleChannel,
                input_resolution = ( 224//w, 224//h),
                window_size = w,
                overlap_ratio = 0.5,
                num_heads = numHead,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio = 2,
                norm_layer=nn.LayerNorm,
            )
        
        self.conv1  =  nn.Sequential(
            nn.Conv2d( inChannel, middleChannel, kernel_size = 3, padding = 1, stride = 1 ),
            GNSiLU(middleChannel, groupHead = numHead, nonlinearity = True)
        )
        self.conv2  =  nn.Sequential(
            nn.Conv2d(middleChannel, ouChannel, kernel_size = 3,padding = 1, stride = 1),
            GNSiLU(ouChannel, groupHead = numHead, nonlinearity = False, init_zero = True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inChannel, ouChannel, 1, 1)
        )
        self.last_act = nn.ReLU()
    
    def forward(self, x):
       
        shortcut  =  x
        B, C, H, W = x.shape 

        # windows fit
        out = x.reshape(B * (H//self.h) * (W//self.w), C, self.h, self.w)
        out = self.conv1(out)
        out = out.permute( 0, 2, 3, 1)  # BCHW -> B,HW,C
        out = out.reshape(B, H*W, -1)
        
        out = self.ocatn(  out, ( H, W), self.rpiTable) # B HW C

        out = out.reshape(B * (H//self.h) * (W//self.w), self.h, self.w, -1)
        out = out.permute(0, 3, 1, 2)

        out  =  self.conv2(out)
        # print(out.shape)
        N1, C1, H1, W1  =  out.shape
        out  =  out.reshape(B, C1, int(H), int(W))

        out +=  self.conv3(shortcut)
        out  =  self.last_act(out)
        return out

from .groupformer import _make_bot_layer
class ATGFomer(nn.Module):
    def __init__(self, inChannel, ouChannel, w = 4, h = 4) -> None:
        super().__init__()
        self.gformer = _make_bot_layer(
            inChannel, ouChannel, w = w,
        )
    
    def forward(self, x):
        out = self.gformer(x)
        return out

class FSM(nn.Module):
    def __init__(self, inChannel, ouChannel, w, h, r = 2, mult = 1, dropout = 0.3, Dtype= 'OCA' ) -> None:
        super().__init__()
        self.inChannel = inChannel
        self.ouChannel = ouChannel
        self.former =  nn.Sequential(
            ATGFomer( inChannel, ouChannel, w )
        ) if Dtype == 'GF' else nn.Sequential(
            ATOCA( inChannel, ouChannel, r, w = w, h = h,
            )
        )
        self.conv = nn.Sequential(
            Level1Waveblock(mult = mult, ff_channel = inChannel, final_dim = inChannel, dropout = dropout),
        )
        # self.conv = nn.Sequential(
        #     CBLK(inChannel,ouChannel),
        #     CBLK(ouChannel,ouChannel),
        #     CBLK(ouChannel,ouChannel,1,1,0),
        # )
        self.sk = SKNConv([inChannel, ouChannel], ouChannel, 8, 2)

    def forward(self, x):
        xConv = self.conv(x)
        xFormer = self.former(x)
        feature = self.sk([xConv, xFormer])
        return feature

class FSMD(nn.Module):
    def __init__(self, inChannel, ouChannel, w, h, Dtype = 'OCA'):
        super(FSMD, self).__init__()
        dimIn = inChannel
        dimOut = ouChannel
        self.conv = FSM( dimIn, dimOut, w, h, Dtype = Dtype)
        print("res g former")
        self.up = nn.Upsample( scale_factor = 2)
        
    def forward(self, x, y):
        y = self.up(y)
        x = torch.cat([x,y], 1)
        x = self.conv(x)
        return x

# u baseline     
class MPW(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        featureList = [1,2,3,4,5,6]
        self.downSample = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.enc_len = 5
        self.enc = nn.ModuleList()
        self.brige = nn.Sequential(
            nn.Conv2d(featureList[-1], featureList[-1], 1,1),
            nn.LayerNorm([featureList[-1], 7, 7]),
            nn.LeakyReLU(inplace=True),
        )
        for i in range(self.enc_len):
            self.enc.append(CBLK(featureList[i], featureList[i+1]))
        self.up = nn.Upsample(scale_factor = 2)
            
        self.dec = nn.ModuleList()
        for i in range(self.enc_len):
            nextChannel = featureList[i]
            nowChannel = featureList[i+1]
            self.dec.append(
                Decode( 2*nowChannel + 1 , nextChannel)
            )
        self.last_up = torch.nn.PixelShuffle(2)
            
    @torch.no_grad()
    def buildPyramid(self, x, pn = 5):
        
        x_list = []
        for i in range(pn):
            x = self.downSample(x)
            #if prompt is not None:
            #     prompt = self.downSample(prompt)
            #     x = x + prompt
            x_list.append(x)
            
        return x_list
    
    def forward(self, x, prompt = None):
        if prompt is None:
            x_list = self.buildPyramid(x)
        else:
            x_list = self.buildPyramid(prompt)
            
        skip_list = []
        for i in range(self.enc_len):
            x = self.enc[i](x)
            skip_list.append(x)
            x = self.maxpool(x)
            # print(f"enc {i} : {x.shape}")
            
        x = self.brige(x)
        #　x = torch.cat([x, skip_list[re_index], x_list[-1]], 1)
        for i in range(self.enc_len):
            re_index = self.enc_len - i - 1
            x = torch.cat([x_list[re_index], x], 1)
            x = self.dec[re_index](x, skip_list[re_index])
        # out = self.last_up(x)
        return x

# sample in Wave and Overlap former
class SWOV1(nn.Module):
    def __init__(self, inChannel = 1, enc_len = 5, featureList = [ 48, 48, 48, 48, 48], needUp = 0) -> None:
        super().__init__()
        featureList = [inChannel, *featureList]
        self.downSample = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.enc_len = enc_len
        self.enc = nn.ModuleList()
        self.brige = nn.Sequential(
            nn.Conv2d(featureList[-1], featureList[-1] , 1,1),
            nn.LayerNorm([featureList[-1], 8, 8]),
            nn.LeakyReLU(inplace=True),
        )
        for i in range(self.enc_len):
            self.enc.append(CBLK(featureList[i], featureList[i+1]))
            
        self.up = nn.Upsample(scale_factor = 2)
            
        self.dec = nn.ModuleList()
        self.embed = nn.ModuleList()
        for i in range(self.enc_len):
            
            nextChannel = featureList[i]
            nowChannel = featureList[i+1] 
            if nextChannel == 1:
                nextChannel = featureList[1]
            self.embed.append(
                CBLK(1, nowChannel, k = 1, s = 1, p = 0)
            )
            self.dec.append(
                FSMD( 2 * nowChannel , nextChannel, w = 4, h = 4, Dtype = "OCA")
            )
        if needUp:
            self.last_up = nn.Sequential(
                Upsample(2 ** needUp, featureList[1]), # 2倍上采样 -> 2^n, super = 64
                nn.Conv2d(featureList[1], featureList[0], 1, 1)
            )
        else:
            self.last_up = nn.Sequential(
                nn.Conv2d(featureList[1], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, inChannel, 1, 1)
            )
        self.act = nn.LeakyReLU(inplace=True)
            
    @torch.no_grad()
    def buildPyramid(self, x, pn = 5):
        x_list = [x]
        for i in range(pn):
            x = self.downSample(x)
            x_list.append(x)
        return x_list
    
    def forward(self, x, prompt = None):
        if prompt is None:
            x_list = self.buildPyramid(x)
        else:
            x_list = self.buildPyramid(prompt)
        
        skip_list = []
        for i in range(self.enc_len):
            x = self.enc[i]( self.act(x + x_list[i])) 
            skip_list.append(x)
            x = self.maxpool(x)
         
            
        x = self.brige(x)
        
        for i in range(self.enc_len):
            re_index = self.enc_len - i - 1
            x =  self.up(x) + self.dec[re_index](skip_list[re_index], x) # 降采样之后上采样避免了噪声的滞留

        x = self.last_up(self.act(x))
        return x

class SWO(nn.Module):
    def __init__(self, inChannel = 1, enc_len = 5, featureList = [ 48, 96, 192, 384, 768], needUp = 0) -> None:
        super().__init__()
        featureList = [inChannel, *featureList]
        self.downSample = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.enc_len = enc_len
        self.enc = nn.ModuleList()
    
        self.brige = nn.Sequential(
            nn.Conv2d(featureList[-1], featureList[-1] , 1,1),
            nn.LayerNorm([featureList[-1], 8, 8]),
            nn.LeakyReLU(inplace=True),
        )
        for i in range(self.enc_len):
            self.enc.append(nn.Sequential(
                CBLK(featureList[i], featureList[i+1]),
                CBLK(featureList[i+1], featureList[i+1]),
                CBLK(featureList[i+1], featureList[i+1]),
            ))

            
        self.up = nn.Upsample(scale_factor = 2)
            
        self.dec = nn.ModuleList()
        self.embed = nn.ModuleList()
        for i in range(self.enc_len):
            
            nextChannel = featureList[i]
            nowChannel = featureList[i+1] 
            if nextChannel == 1:
                nextChannel = featureList[1]
            self.embed.append(
                CBLK(1, nextChannel),
            )
            self.dec.append(
                FSMD( 2 * nowChannel , nextChannel, w = 4, h = 4, Dtype = "OCA")
            )
        if needUp:
            self.last_up = nn.Sequential(
                Upsample(2 ** needUp, featureList[1]), # 2倍上采样 -> 2^n, super = 64
                nn.Conv2d(featureList[1], featureList[0], 1, 1),
            )
        else:
            self.last_up = nn.Sequential(
                nn.Conv2d(featureList[1], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 32, 1, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, inChannel, 1, 1)
            )
        self.act = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax()
 
    @torch.no_grad()
    def buildPyramid(self, x, pn = 5):
        x_list = [x]
        for i in range(pn):
            x = self.downSample(x)
            x_list.append(x)
        return x_list
    
    def skatn(self, x):
        x = torch.sum(x, dim=1, keepdim=True)
        atn = self.softmax(x)
        return atn
    
    def forward(self, x, prompt = None):
        if prompt is None:
            x_list = self.buildPyramid(x)
        else:
            x_list = self.buildPyramid(prompt)
        
        skip_list = []
        denoise_list = []
        
        for i in range(self.enc_len):
           
            x = self.enc[i](x) 
            denoise_attn =  self.skatn(x)
            denoise = denoise_attn * x_list[i]
           
            skip_list.append(x)
            denoise_list.append(denoise)
            x = self.maxpool(x)
         
        x = self.brige(x)
        
        for i in range(self.enc_len):
            re_index = self.enc_len - i - 1
            x = self.embed[re_index]( denoise_list[re_index]) + self.dec[re_index]( skip_list[re_index], x)  # 降采样之后上采样避免了噪声的滞留

        x = self.last_up(self.act(x))
        return x  
     
class SWGFV1(nn.Module):
    def __init__(self, inChannel = 1, enc_len = 5, featureList = [ 48, 48, 48, 48, 48], needUp = 0) -> None:
        super().__init__()
        featureList = [inChannel, *featureList]
        self.downSample = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor = 2)

        self.enc_len = enc_len
        self.brige = nn.Sequential(
            nn.Conv2d(inChannel, featureList[1], 2, 2),
            Level1Waveblock( mult = 1, ff_channel = featureList[1], final_dim = featureList[1], dropout = 0),
            nn.Conv2d(featureList[1], featureList[1], 16, 16),
            nn.LayerNorm([featureList[1], 8, 8]),
            nn.LeakyReLU(inplace=True),
            )
        self.dec = nn.ModuleList()
        self.embed = nn.ModuleList()
        self.non = nn.ModuleList()
        for i in range(self.enc_len):
            nextChannel = featureList[i]
            nowChannel = featureList[i+1] 
            if nextChannel == 1:
                nextChannel = featureList[1]
            print("FSMD:", 2 * nowChannel, nextChannel)
            self.embed.append(
                CBLK(1, nowChannel, k = 3, s = 1, p = 1)
            )
            self.dec.append(
                FSMD( 2 * nowChannel , nextChannel, w = 4, h = 4, Dtype = "OCA")
            )
            
            self.non.append(
                NonLocalBlockM(nextChannel) if i >= 2 else SpatialAttention(3)
            )
        if needUp:
            self.last_up = nn.Sequential(
                Upsample(2 ** needUp, featureList[1]), # 2倍上采样 -> 2^n, super = 64
                nn.Conv2d(featureList[1], featureList[0], 1, 1)
            )
        else:
            self.last_up = nn.Sequential(
                CBLK(featureList[1], 64, k = 3, s = 1, p = 1),
                CBLK(64, 64, k = 1, s = 1, p = 0),
                nn.Conv2d(64, featureList[0], 1, 1 ),
            )
        self.act = nn.LeakyReLU(inplace=True)
            
    @torch.no_grad()
    def buildPyramid(self, x, pn = 5):
        x_list = [x]
        for i in range(pn):
            x = self.downSample(x)
            x_list.append(x)
        return x_list
    
    def forward(self, x, prompt = None):
        if prompt is None:
            x_list = self.buildPyramid(x)
        else:
            x_list = self.buildPyramid(prompt)
            
        xBrige = self.brige(x)
        for i in x_list:
            print(i.shape)
        # print("middle:", xBrige.shape)
        for i in range(self.enc_len):
            re_index = self.enc_len - i - 1
            x_p = self.embed[re_index](x_list[re_index])
            up_brige = self.up(xBrige)
            xBrige = self.dec[re_index](x_p, xBrige)
            if i < 3:
                # print(re_index)
                xBrige = up_brige + self.non[re_index](xBrige)
            else:
                # print(re_index, "sp", xBrige.shape)
                xBrige = xBrige * self.non[re_index](xBrige)
            # print("dec: ", xBrige.shape)
        x = xBrige
        x = self.last_up(self.act(x))
        return x
    
class SWGF(nn.Module):
    def __init__(self, inChannel = 1, enc_len = 5, featureList = [ 48, 48, 48, 48, 48], needUp = 0) -> None:
        super().__init__()
        featureList = [inChannel, *featureList]
        self.downSample = nn.AvgPool2d(2)
        self.maxpool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor = 2)
        
        waveBlockChannel = 144
        self.enc_len = enc_len
        self.brige = nn.Sequential(
            nn.Conv2d(1, waveBlockChannel, 2, 2),
            Level1Waveblock( mult = 1, ff_channel = waveBlockChannel, final_dim = waveBlockChannel, dropout = 0),
            nn.Conv2d(waveBlockChannel, waveBlockChannel , 7, 4, 3),
            nn.Conv2d(waveBlockChannel, 2 * waveBlockChannel , 4, 2, 1),
            nn.Conv2d(2 * waveBlockChannel, featureList[-1] , 2, 2),
            nn.LayerNorm([featureList[-1], 8, 8]),
            nn.LeakyReLU(inplace=True),
            )
        self.dec = nn.ModuleList()
        self.embed = nn.ModuleList()

        
        for i in range(self.enc_len):
            nextChannel = featureList[i]
            nowChannel = featureList[i+1] 
            if nextChannel == 1:
                nextChannel = featureList[1]//2
            print("FSMD:", 2 * nowChannel, nextChannel)
            self.embed.append(
                SMA( 1, nowChannel),
                # CBLK(1, nowChannel, k = 1, s = 1, p = 0),
            )
            self.dec.append(
                FSMD( 2 * nowChannel , nextChannel, w = 4, h = 4, Dtype = "OCA")
            )
            
        if needUp:
            self.last_up = nn.Sequential(
                Upsample(2 ** needUp, featureList[1]//2), # 2倍上采样 -> 2^n, super = 64
                nn.Conv2d(featureList[1]//2, featureList[0], 1, 1)
            )
        else:
            self.last_up = nn.Sequential(
                CBLK(featureList[1]//2, 64, k = 3, s = 1, p = 1),
                CBLK(64, 64, k = 1, s = 1, p = 0),
                nn.Conv2d(64, featureList[0], 1, 1 ),
            )
        self.act = nn.LeakyReLU(inplace=True)
            
    @torch.no_grad()
    def buildPyramid(self, x, pn = 5):
        x_list = [x]
        for i in range(pn):
            x = self.downSample(x)
            x_list.append(x)
        return x_list
    
    def easyDown(self, x):
        y = []
        entropy = torch.mean(torch.mean(x, -1), -1)
        _,index = torch.sort(entropy)
        for batchIndex in range(len(index)):
            feature = x[batchIndex, index[batchIndex], :, :]
            Odd = feature[::2,:,:]
            Even = feature[1::2,:,:]
            featureSum = (Odd + Even)
            y.append(featureSum)
        y = torch.stack(y,0)
        return y
    
    def forward(self, x, prompt = None):
        if prompt is None:
            x_list = self.buildPyramid(x)
        else:
            x_list = self.buildPyramid(prompt)
        
        x = self.brige(x)
        # print("brige:", x.shape)
        for i in range(self.enc_len):
            re_index = self.enc_len - i - 1
            fpn_feature = self.embed[re_index]( x_list[re_index])
            #print(i)
            x = self.dec[re_index]( fpn_feature, x)
            x = x + ( self.easyDown(fpn_feature) if x.shape[1] != fpn_feature.shape[1] else fpn_feature)   # 降采样之后上采样避免了噪声的滞留
            
                
        x = self.last_up(self.act(x))
        return x