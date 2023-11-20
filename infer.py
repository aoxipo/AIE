from dataloader import DataLoad, DataloadTest
from train import Train
import os
import torch
import GPUtil
from utils.logger import Logger
import torch
import csv
import sys
import os
import numpy as np

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
    ,"HypoF_Y","CNV","Vascular abnormality (DR)","Pattern", "ID", "Folder"]

def query_name(ans, a):
    infor_list = []
    for i in range(14): # 0 - 13
        name_id = i # 0- 13
        #print(ans[name_id].squeeze().cpu().numpy())
        #print(np.argmax(ans[name_id].squeeze().cpu().numpy()))
        infor_list.append( a.gt_index2name_dict[name_id][np.argmax(ans[name_id].squeeze().cpu().numpy())] )
    return infor_list

def infer_once(model, image_path, dataloader):
    image = dataloader.read_image(image_path)
    image_tensor = dataloader.datagan(image)
    image_tensor = image_tensor.unsqueeze(0)
    ans = model.predict_batch(image_tensor)
    return query_name(ans, dataloader) # 返回查询名称


if __name__ == "__main__":
    image_shape = (256,256)
    root_path = r"D:\dataset\eye\Train" # 用于获取标签
    val_path = r"D:\dataset\eye\Train\val/"
    # 日志准备
    logger = Logger(
        file_name = "log_infer.txt",
        file_mode= "w+",#"a+",
        should_flush=True
    )
    # 训练模式准备
    trainer = Train( 
        1, 256,
        name = "aie",
        method_type = 1,
        batch_size = 20,
        device_ = "cuda:0",
        is_show = False,
    )

    # 加载你的参数路径
    #trainer.load_parameter(r"./save_best/aie/best.pkl")
    
    # 数据集标签获取
    a = DataLoad(root_path, image_shape = image_shape)
    a.set_gan()
    # 数据加载
    dataset = DataloadTest(val_path)
    print("total load :", len(dataset))
    total = len(dataset)
    # 循环读取
    mark_dict = {}
    with open('model_results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data_keys)  # 写入表头
        index = 0
        for eye_object in dataset.photo_set:
            index += 1
            file_path = eye_object["path"]
            folder = eye_object["folder"]
            eye_id = folder.split("_")[0]
            name = eye_object["name"]
            if folder in mark_dict:
                continue
            result = infer_once(trainer, file_path, a) # a 这个地方是为了获取标签 病症的描述信息
            result.append( str(eye_id) )
            result.append( folder )
            writer.writerow(result)  # 写入结果数据
            mark_dict[folder] = 1
            if index % 100:
                print( "process: {}/{}".format( index, total) )
            
    print("overed")

    



    
    


    # 查询 单次结果
    # name_id = 2 # 0- 13
    # class_name = a.data_key[name_id]
    # #获取name_id下的这一列的特征
    # # print(f"class name: {class_name}, info: {a.gt_index2name_dict[name_id][np.argmax(gt[name_id]).item()]}")
    
    