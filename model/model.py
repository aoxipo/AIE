import torch
import torch.nn as nn

class BaseLine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        pass

    def build_result(self, x):
        return { "pred":x }
    

class Neck(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

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

from .pvtv2 import pvt_v2_b2
from torchvision.models import resnet34 as resnet
class DUAL(BaseLine):
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
    def __init__(self) -> None:
        super().__init__()
        
        
        # PVT 提取特征
        path = './pretrained_pth/pvt_v2_b2.pth' # 找我要
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        #  save_model = torch.load(path)
        #         model_dict = self.backbone.state_dict()
        #         state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        #         model_dict.update(state_dict)
        #         self.backbone.load_state_dict(model_dict)
        #         n_p = sum(x.numel() for x in self.backbone.parameters()) # number parameters
        #         n_g = sum(x.numel() for x in self.backbone.parameters() if x.requires_grad)  # number gradients
        #         print(f"pvt Summary: {len(list(self.backbone.modules()))} layers, {n_p} parameters, {n_p/1e6} M, {n_g} gradients")
        # RESNET 特征提取
        self.resnet = resnet(pretrained=True) 
        # self.resnet.load_state_dict(torch.load('pretrained_pth/resnet34-43635321.pth')) # 找我要
        n_p = sum(x.numel() for x in self.resnet.parameters()) # number parameters
        n_g = sum(x.numel() for x in self.resnet.parameters() if x.requires_grad)  # number gradients
        print(f"ResNet Summary: {len(list(self.resnet.modules()))} layers, {n_p} parameters, {n_p/1e6} M, {n_g} gradients")
        self.head = Head(middle_channel=[])

    def pvt_backbone(self, x):
        pvt_x = x.clone().detach()
        if x.shape[1] == 1:
            pvt_x = torch.cat([pvt_x,pvt_x,pvt_x], 1)

        pvt = self.backbone(pvt_x)
        # pvt_decode: x1:torch.Size([2, 64, 56, 56]), c2:torch.Size([2, 128, 28, 28]), c3:torch.Size([2, 320, 14, 14]), c4:torch.Size([2, 512, 7, 7])
        
        return pvt

    def resnet_backbone(self, x):
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
        return x1,x2, x3,x4
    def forward(self, x):
        pvt_decode = self.pvt_backbone(x)     
        res_decode = self.resnet_backbone(x)
        
        return None