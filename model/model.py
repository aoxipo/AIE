import torch
import torch.nn as nn

class BaseLine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass

    def build_result(self, x):
        return { "pred":x }
class Block(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class CBLK(nn.Module):
    def __init__(self, inc, ouc, k = 3, s = 1, p = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, ouc, k, s, p),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x
# sppf  atn
class Fusion(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        
        d = ouc//4
        print("inc {}".format(inc))
        self.c1 = nn.Sequential(
            CBLK(inc[0], d),
            nn.MaxPool2d(2),
        )
        self.c2 = nn.Sequential(
            CBLK(inc[1], d)
        )
        self.c3 = nn.Sequential(
            CBLK(inc[2], d)
        )
        self.c4 = nn.Sequential(
            CBLK(inc[3], d)
        )
        self.proj_k = nn.Conv2d(4*d, 4*d, 1,1 )
        self.out = nn.Conv2d(4*d, ouc, 1,1 )
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(0.3)
        self.down = nn.MaxPool2d(2)
        
    def forward(self, x1, x2, x3, x4):
        c1 = self.c1(x1)
        c2 = self.c2(x2)
        c3 = self.c3(x3)
        c4 = self.c4(x4)
        
        c21 = torch.cat([c1,c2], 1)
        # print("c3 {} self.down(c21) {} ".format(c3.size(),self.down(c21).size()))
        
        # print("c1 {} c2 {} c3 {} c4 {} ".format(c1.size(),c2.size(),c3.size(),c4.size() ))
#      c3 torch.Size([2, 36, 14, 14]) self.down(c21) torch.Size([2, 72, 14, 14]) 
# c1 torch.Size([2, 36, 28, 28]) c2 torch.Size([2, 36, 28, 28]) c3 torch.Size([2, 36, 14, 14]) c4 torch.Size([2, 36, 7, 7])    

        c321 = torch.cat([self.down(c21), c3],dim=1)
        
        v = torch.cat([self.down(c321), c4],dim=1)
        k = self.proj_k(v)
        q = self.softmax(torch.sum((self.down(self.down(c1 + c2) + c3) + c4), dim=1, keepdim=True))
        
        x = v @ self.drop(k)
        feature = x @ q
        out = self.out(feature)
        return out
#  多目标 对齐 特征 时序 特征长短不一 优化
#　基于双编码不定长时序眼球诊断优化算法
import numpy as np
class Neck(nn.Module): 
    def __init__(self, pvt_decode, resnet_decode, num_conv_layers = [ 8, 6]):
        super(Neck, self).__init__()
        self.pvt_decode = pvt_decode
        self.resnet_decode = resnet_decode
        middle_channel = 144
        last_channel = 32
        print("inc {}".format(np.array(pvt_decode) + np.array(resnet_decode)))
        self.focus = Fusion( np.array(pvt_decode) + np.array(resnet_decode), middle_channel)
      
        d = [ CBLK(middle_channel, middle_channel,1,1,0) for i in range(num_conv_layers[0]) ]
        d.append(CBLK(middle_channel, last_channel ,1,1,0))
        self.conv_layers1 = nn.Sequential(*d)
        
        d = [ CBLK(middle_channel, middle_channel,1,1,0) for i in range(num_conv_layers[1]) ]
        d.append(CBLK(middle_channel, last_channel ,1,1,0))
        self.conv_layers2 = nn.Sequential(*d)
        self.conv = nn.ModuleList()
        self.flatten = nn.Flatten()
        self.linear=nn.ModuleList()
        
        for i in range(14):
            self.linear.append(
                nn.Sequential(
                    nn.Linear( 2048, 1024),# last_channel*7*7 middle_channel*7*7, 1024),
                    nn.Linear(1024, 768),
                    nn.Linear(768, 512)
                )
            )
           
    
    def forward(self, x, y):
        fuse1 = torch.cat([x[0] , y[0]], 1)
        fuse2 = torch.cat([x[1] , y[1]], 1)
        fuse3 = torch.cat([x[2] , y[2]], 1)
        fuse4 = torch.cat([x[3] , y[3]], 1)
        # print("fuse1 {} fuse2 {} fuse3 {} fuse4 {} ".format(fuse1.size(),fuse2.size(),fuse3.size(),fuse4.size()))
        combine_feature = self.focus( fuse1,fuse2,fuse3,fuse4 )
        # print("combine feature:", combine_feature.shape)
        out1 = self.conv_layers1(combine_feature)
        out2 = self.conv_layers2(combine_feature)
        high = self.flatten(out1)
        low = self.flatten(out2)
        # print(high.shape, low.shape)
        ans = []
        for idx in range(14):
            # print(high.shape, low.shape)
            if idx < 8:
                ans.append(self.linear[idx](high))
            else:
                ans.append(self.linear[idx](low))
            
        return ans

class Head(nn.Module):
    # 类别数量
    Impression = 23 # 0
    HyperF_Type = 5 # 1
    HyperF_Area = 3 # 2
    HyperF_Fovea = 2 # 3
    HyperF_ExtraFovea = 18 # 4
    HyperF_Y = 4 #           5  
    HypoF_Type = 3 #         6 
    HypoF_Area = 3 #         7
    HypoF_Fovea = 2 #        8
    HypoF_ExtraFovea = 17 #  9
    HypoF_Y = 5           #  10
    CNV = 2               #  11
    Vascular_abnormality = 15 # 12
    Pattern = 14              # 13
    # [0,1,6,7,9,10,11,13]
    # [2,3,4,5,8,12]
    def __init__(self, middle_channel):
        super().__init__()
        self.Impression_classifier = nn.Sequential(nn.Linear(middle_channel[0], self.Impression))
        self.HyperF_Type_classifier = nn.Sequential(nn.Linear(middle_channel[1], self.HyperF_Type))
        self.HyperF_Area_classifier = nn.Sequential(nn.Linear(middle_channel[2], self.HyperF_Area))
        self.HyperF_Fovea_classifier = nn.Sequential(nn.Linear(middle_channel[3], self.HyperF_Fovea))
        self.HyperF_ExtraFovea_classifier = nn.Sequential(nn.Linear(middle_channel[4], self.HyperF_ExtraFovea))
        self.HyperF_Y_classifier = nn.Sequential(nn.Linear(middle_channel[5], self.HyperF_Y))
        self.HypoF_Type_classifier = nn.Sequential(nn.Linear(middle_channel[6], self.HypoF_Type))
        self.HypoF_Area_classifier = nn.Sequential(nn.Linear(middle_channel[7], self.HypoF_Area))
        self.HypoF_Fovea_classifier = nn.Sequential(nn.Linear(middle_channel[8], self.HypoF_Fovea))
        self.HypoF_ExtraFovea_classifier = nn.Sequential(nn.Linear(middle_channel[9], self.HypoF_ExtraFovea))
        self.HypoF_Y_classifier = nn.Sequential(nn.Linear(middle_channel[10], self.HypoF_Y))
        self.CNV_classifier = nn.Sequential(nn.Linear(middle_channel[11], self.CNV))
        self.Vascular_abnormality_classifier = nn.Sequential(nn.Linear(middle_channel[12], self.Vascular_abnormality))
        self.Pattern_classifier = nn.Sequential(nn.Linear(middle_channel[13], self.Pattern))
        
    def forward(self, x):
        Impression_res = self.Impression_classifier(x[0])
        HyperF_Type_res = self.HyperF_Type_classifier(x[1])
        HyperF_Area_res = self.HyperF_Area_classifier(x[2])
        HyperF_Fovea_res = self.HyperF_Fovea_classifier(x[3])
        HyperF_ExtraFovea_res = self.HyperF_ExtraFovea_classifier(x[4])
        HyperF_Y_res = self.HyperF_Y_classifier(x[5])
        HypoF_Type_res = self.HypoF_Type_classifier(x[6])
        HypoF_Area_res = self.HypoF_Area_classifier(x[7])
        HypoF_Fovea_res = self.HypoF_Fovea_classifier(x[8])
        HypoF_ExtraFovea_res = self.HypoF_ExtraFovea_classifier(x[9])
        HypoF_Y_res = self.HypoF_Y_classifier(x[10])
        CNV_res = self.CNV_classifier(x[11])
        Vascular_abnormality_res = self.Vascular_abnormality_classifier(x[12])
        Pattern_res = self.Pattern_classifier(x[13])
        return  [
            Impression_res, HyperF_Type_res, HyperF_Area_res, HyperF_Fovea_res, HyperF_ExtraFovea_res, 
            HyperF_Y_res, HypoF_Type_res, HypoF_Area_res, HypoF_Fovea_res, 
            HypoF_ExtraFovea_res, HypoF_Y_res, CNV_res, Vascular_abnormality_res, 
            Pattern_res 
        ]


class AFC(nn.Module):            
    def __init__(self, features_list, out_features, r = 2, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            r: the radio for compute d, the length of z.                 2      
            L: the minimum dim of the vector z in paper, default 32
        """
        super(AFC, self).__init__()
        features = out_features
        d = max(int(features/r), L)
        self.M = len(features_list)
        self.features = features
        self.convs = nn.ModuleList()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList()
        for i in range(self.M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1) 
        
    def normal(self, x):
        return (x - x.min())/(x.max() - x.min())
    
    def forward(self, x):
        total = len(x) 
        for i in range(total):
            fea = x[i].unsqueeze_(dim = 0).unsqueeze_(dim = 1)
            # print(fea.shape)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        # print(feas.shape)
        fea_U = torch.sum(feas, dim=1)#.unsqueeze(0)
        # print("feau:", fea_U.shape)
        fea_s = self.gap(fea_U).squeeze(dim=-1).squeeze(dim=-1)
        # print(fea_s.shape, self.fc)
        fea_z = self.fc(fea_s)
        # print( fea_z.shape )
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
           
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = self.normal(feas.sum(dim=1)) * self.normal(attention_vectors.sum(dim=1).squeeze(1))
        return fea_v
    

from .pvtv2 import pvt_v2_b2
from torchvision.models import resnet34 as resnet
class DUAL(BaseLine): #pipline
    # 类别数量
    Impression = 23 # 0
    HyperF_Type = 5 # 1
    HyperF_Area = 3 # 2
    HyperF_Fovea = 2 # 3
    HyperF_ExtraFovea = 18 # 4
    HyperF_Y = 4 #           5  
    HypoF_Type = 3 #         6 
    HypoF_Area = 3 #         7
    HypoF_Fovea = 2 #        8
    HypoF_ExtraFovea = 17 #  9
    HypoF_Y = 5           #  10
    CNV = 2               #  11
    Vascular_abnormality = 15 # 12
    Pattern = 14              # 13
    
    # [0,1,6,7,9,10,11,13]
    
    # [2,3,4,5,8,12]
    def __init__(self, neck_num = [8,6]) -> None:
        super().__init__()
        
        # PVT 提取特征
        path = './model/pretrained_pth/pvt_v2_b2.pth' # 找我要
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        n_p = sum(x.numel() for x in self.backbone.parameters()) # number parameters
        n_g = sum(x.numel() for x in self.backbone.parameters() if x.requires_grad)  # number gradients
        print(f"pvt Summary: {len(list(self.backbone.modules()))} layers, {n_p} parameters, {n_p/1e6} M, {n_g} gradients")
        # RESNET 特征提取
        self.resnet = resnet(pretrained=True) 
        # self.resnet.load_state_dict(torch.load('pretrained_pth/resnet34-43635321.pth')) # 找我要
        n_p = sum(x.numel() for x in self.resnet.parameters()) # number parameters
        n_g = sum(x.numel() for x in self.resnet.parameters() if x.requires_grad)  # number gradients
        print(f"ResNet Summary: {len(list(self.resnet.modules()))} layers, {n_p} parameters, {n_p/1e6} M, {n_g} gradients")
#         self.neck = Neck( pvt_feature = [64, 128, 320, 512], resnet_feature = [64, 128, 256, 512], num_conv_layers = neck_num)
        pvt_feature = [64, 128, 320, 512]
        resnet_feature = [64, 128, 256, 512]
        self.neck = Neck( pvt_feature, resnet_feature, num_conv_layers = neck_num)
        self.head = Head(middle_channel=[ 512 for i in range(14) ])
        self.afc = nn.ModuleList()
        self.afc1 = nn.ModuleList()
        for i in range(4):
            self.afc.append( 
                AFC([ pvt_feature[i] for ii in range(4)], pvt_feature[i])
            )
            self.afc1.append( 
                AFC([ resnet_feature[i] for ii in range(4)], resnet_feature[i])
            )
        
    
    @torch.no_grad()
    def pvt_backbone(self, x):
        pvt_x = x # .clone().detach()
        pvt_x = torch.stack(pvt_x, 0)
        # if x.shape[1] == 1:
        #     pvt_x = torch.cat([pvt_x,pvt_x,pvt_x], 1)

        pvt = self.backbone(pvt_x)
        # pvt_decode: x1:torch.Size([2, 64, 56, 56]), c2:torch.Size([2, 128, 28, 28]), c3:torch.Size([2, 320, 14, 14]), c4:torch.Size([2, 512, 7, 7])
        return pvt
    
    @torch.no_grad()
    def resnet_backbone(self, x):
        # if x.shape[1] == 1:
        #     x = torch.cat([x,x,x], 1)
        x = torch.stack(x, 0)
        x   = self.resnet.conv1(x)
        x   = self.resnet.bn1(x)
        x   = self.resnet.relu(x)
        
        # - low-level features
        x0  = self.resnet.maxpool(x)       
        x1  = self.resnet.layer1(x0)       
        x2  = self.resnet.layer2(x1)       
        x3  = self.resnet.layer3(x2)     
        x4  = self.resnet.layer4(x3)     
        #res: x1:torch.Size([2, 64, 56, 56]), c2:torch.Size([2, 128, 28, 28]), c3:torch.Size([2, 256, 14, 14]), c4:torch.Size([2, 512, 7, 7])
        #print(f"res:x:{x.shape}, x1:{x1.shape}, c2:{x2.shape}, c3:{x3.shape}, c4:{x4.shape}")
        # print(f"pvt_decode:x:{x.shape}, x1:{pvt_decode[0].shape}, c2:{pvt_decode[1].shape}, c3:{pvt_decode[2].shape}, c4:{pvt_decode[3].shape}")
        return x1, x2, x3,x4
    def forward(self, x_list):
        pvt_decode_list = [[],[],[],[]]
        res_decode_list = [[],[],[],[]]
        # print("encode:", len(x_list))
        for x in x_list:
            # x = x.unsqueeze(0)
            pvt_decode = self.pvt_backbone(x)     
            res_decode = self.resnet_backbone(x)
            for i in range(4):
                # print( pvt_decode[i].shape, res_decode[i].shape )
                pvt_decode_list[i].append( pvt_decode[i] )
                res_decode_list[i].append( res_decode[i] )

        pvt_decode = [[],[],[],[]]
        res_decode = [[],[],[],[]]

        for i in range(len(x_list)):
            # print(len(pvt_decode_list[0]))
            # print(pvt_decode_list[0][i].shape)
            for j in range(4):
                pvt_feature = self.afc[j](pvt_decode_list[j][i])
                res_feature = self.afc1[j](res_decode_list[j][i])
                pvt_decode[j].append(pvt_feature)
                res_decode[j].append(res_feature)

        for i in range(4):
            pvt_decode[i] = torch.cat(pvt_decode[i] , 0)
            res_decode[i] = torch.cat(res_decode[i] , 0)
            # print(pvt_decode[i].shape, res_decode[i].shape)

        feature_neck = self.neck( pvt_decode, res_decode)
        
        classifier = self.head(feature_neck)
        return classifier