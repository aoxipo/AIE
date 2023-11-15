import sys
sys.path.append('C:/Users/csz/Desktop/cs/AIE')
from dataloader import DataLoad
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import os
import torch
import numpy as np
import datetime
import GPUtil
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
from utils.adploss import diceCoeffv2
from utils.score import cal_all_score
from utils.score  import mean_iou_np, mean_dice_np, positive_recall, negative_recall
from utils.show import display_progress
from utils.logger import Logger
use_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(3407)
if (use_gpu):
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    if (len(deviceIDs) != 0):
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=1, maxMemory=1, includeNan=False, excludeID=[],
                                        excludeUUID=[])
        print(deviceIDs)
        print("detect set :", deviceIDs)
        device = torch.device("cuda:" + str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)
data_keys = [ "Impression", "HyperF_Type", "HyperF_Area(DA)", "HyperF_Fovea", "HyperF_ExtraFovea", "HyperF_Y", 
      "HypoF_Type" ,"HypoF_Area(DA)","HypoF_Fovea", "HypoF_ExtraFovea"
    ,"HypoF_Y","CNV","Vascular abnormality (DR)","Pattern"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import sys
import os
import numpy as np
# 将train.py所在的目录添加到sys.path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from dataloader import DataLoad
import train

if __name__ == "__main__":
    trainer = Train( 
            1, 256,
            name = "aie",
            method_type = 1,
            batch_size = 20,
            device_ = "cuda:0",
            is_show = False,
        )

    input_data = torch.randn(2, 3, 256, 256)
    res = trainer.model(input_data)
    root_path = r"F:\dataset\Train"
    image_shape = (256,256)

    a = DataLoad(root_path, image_shape = image_shape)
    a.set_gan()
    print(len(a))


    # 查询
    name_id = 2 # 0- 13
    class_name = a.data_key[name_id]
    #获取name_id下的这一列的特征
    print(f"class name: {class_name}, info: {a.gt_index2name_dict[name_id][np.argmax(gt[name_id]).item()]}")
    
    results = []

    for row in range(res[0].shape[0]):
        curRow = []
        for i in range(len(res)):
            max_index = torch.argmax(res[i])

            # 使用索引获取最大值
            max_value = res[i].flatten()[max_index]
            try:
                curRow.append(a.gt_index2name_dict[i][max_index.item()])
            except Exception as e:
                print("{} 行，第 {} 列 值是 {}".format(row,i,e) )

            # 对应的列名
            # 检查列名和结果数量是否匹配
        results.append(curRow)

    if len(data_keys) != len(results):
        print("列名和结果数量不匹配，请检查数据。{}".format(len(results[0])))

    else:
        # 创建并写入CSV文件
        with open('model_results.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data_keys)  # 写入表头

            writer.writerow(results)  # 写入结果数据

        print("CSV文件已创建并保存")
        print("{} :最大值的索引: {}".format(i, a.gt_index2name_dict[i][max_index.item()]))