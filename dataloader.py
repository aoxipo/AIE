import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import traceback
class CONDITION:
    def __init__(self, name, idx, belong_name, belong_idx):
        self.name = name
        self.idx = idx
        self.belong_name = belong_name
        self.belong_idx = belong_idx

def anylaized(data_source):
    data_key = [ "Impression", "HyperF_Type", "HyperF_Area(DA)", "HyperF_Fovea", "HyperF_ExtraFovea", "HyperF_Y", 
      "HypoF_Type" ,"HypoF_Area(DA)","HypoF_Fovea", "HypoF_ExtraFovea"
    ,"HypoF_Y","CNV","Vascular abnormality (DR)","Pattern"]
    discribe_word = {}
    discribe_key_word = {}
    discribe_key_index = {}
    index = 0
    
    for keyword in data_key:
        discribe_key_index[keyword] = index
        index += 1
        impress_type = {}
        for i in data_source[keyword]:
            #if i is None:
            if i is np.nan:
                i = 'nan'
            word_list = i.split(',')
            for word in word_list:
                if word == '':
                    continue
                if word not in impress_type:
                    impress_type[word] = 1
                else:
                    impress_type[word] += 1
        discribe_key_word[keyword] = impress_type
        for key, value in impress_type.items():
            if key not in discribe_word:
                discribe_word[key] = 1
            else:
                discribe_word[key] += 1
    return discribe_word, discribe_key_word, discribe_key_index

def encode_row(discribe_key_word, discribe_key_index, discribe_word, data_key):
    index = 0
    gt_row_name2index_dict = {}
    for i in data_key:
        gt_row_name2index_dict[i] = index
        index += 1
    gt_dict = {}
    gt_index2name_dict = {}
    key_index = 0
    for key, value in discribe_key_word.items(): 
        gt_dict[key] = {}
        gt_index2name_dict[gt_row_name2index_dict[key]] = {}
        index = 0
        for gt_key_name in value.keys():
            gt_dict[key][gt_key_name] = index
            gt_index2name_dict[gt_row_name2index_dict[key]][index]=gt_key_name 
            discribe_word[gt_key_name] = CONDITION( gt_key_name, index, key, discribe_key_index[key] )
            # print(gt_key_name)
            index += 1
    return gt_dict, gt_index2name_dict, discribe_word

class DataLoad(Dataset):
    def __init__(self, root_path = r"C:\Users\csz\Desktop\cs\Train" , image_shape = (384,384), data_aug = 1) -> None:
        self.root_path = root_path
        csv_path = root_path + "/Train.csv"
        self.image_shape = image_shape
        self.data_source = pd.read_csv(csv_path)
        # print("")
        data_key = [ "Impression", "HyperF_Type", "HyperF_Area(DA)", "HyperF_Fovea", "HyperF_ExtraFovea", "HyperF_Y",
          "HypoF_Type" ,"HypoF_Area(DA)","HypoF_Fovea", "HypoF_ExtraFovea"
        ,"HypoF_Y","CNV","Vascular abnormality (DR)","Pattern"]
        self.data_key=data_key
        discribe_word, discribe_key_word, discribe_key_index = anylaized(self.data_source)
        gt_dict, gt_index2name_dict, discribe_word = encode_row(discribe_key_word, discribe_key_index, discribe_word, data_key)
        self.gt_dict = gt_dict
        self.gt_index2name_dict = gt_index2name_dict
        self.discribe_word = discribe_word
        
        index = 0
        self.data_key2index = {}
        self.data_key_len = {}
        
        for i in data_key:
            self.data_key2index[i] = index
            self.data_key_len[index] = len(self.gt_dict[i])
            index+=1
        
        self.load_data(self.data_source)
        self.one_hot_label_set()
        self.set_gan()
        d = [ self.photo_set_one_hot for i in range(data_aug)]
        self.photo_set_one_hot = np.array(d).flatten()
        
    def check_word(self, discribe):
        if discribe is np.nan:
            discribe = 'nan'
        word_list = discribe.split(',')
        ans = []
        for word in word_list:
            if word == '':
                continue
            ans.append(word)
        return ans
    
    def one_hot_label_set(self):
        self.photo_set_one_hot = []
        for obj in self.photo_set:
            data_key_idx = 0
            # [[1,3], [1], [2], [0], [0], [0], [0], [2], [0], [0], [0], [0], [0], [0]]
            gt_data_key = []
            for data_key_item_gt in obj['gt']:
                total = self.data_key_len[data_key_idx]
                label_one_hot = np.zeros(total)
                
                for msg in data_key_item_gt:
                    label_one_hot[msg] = 1
                gt_data_key.append( label_one_hot )
                data_key_idx += 1 
            
            # 左右两张图
            self.photo_set_one_hot.append({
                'image':obj['image'],
                'gt':obj['gt'],
                'gt_oh':gt_data_key,
                'direct':1,
            })
            self.photo_set_one_hot.append({
                'image':obj['image'],
                'gt':obj['gt'],
                'gt_oh':gt_data_key,
                'direct':0,
            })
        self.total_number = len(self.photo_set_one_hot)

    # TODO
    # def sample_freq_table(self, data_source):
    #     data_dict = {}  #
    #     max_repetition = 5
    #     for index, row in data_source.iterrows():
    #         key = ""
    #         for name in row[:-2]:
    #             key += str(name)
    #             print(key)
    #         if key not in data_dict:
    #             data_dict[key] = [index]
    #         else:
    #             data_dict[key].append(index)
    #
    #     random_list = []
    #     for key, value in data_dict.items():
    #         rep = len(value)
    #         if rep > max_repetition:
    #             random_list += random.sample(value, max_repetition)
    #         else:
    #             random_list += value

    def load_data(self, data_source):
        photo_set = []

        img_root_path = self.root_path + "/Train/"

        #
        data_dict = {}
        max_repetition = 50
        for index, row in data_source.iterrows():
            key = ""
            for name in row[:-2]:
                key += str(name)
                # print(key)
            if key not in data_dict:
                data_dict[key] = [index]
            else:
                data_dict[key].append(index)
        #

        for index, row in data_source.iterrows():

            gt_index = []

            hashkey = ""
            for discribe_list in row[:-2]:
                discribe_list = self.check_word(discribe_list)
                hashkey += str(discribe_list)

                for discribe in discribe_list:
                    temp_gt_idx = []
                    code = self.discribe_word[discribe].idx
                    temp_gt_idx.append( code )

                gt_index.append(temp_gt_idx)
            #
            if hashkey not in data_dict:
                data_dict[hashkey] = [index]
            else:
                data_dict[hashkey].append(index)
                if len(data_dict[hashkey]) > max_repetition:
                    continue
            #
            file_folder = row[-1]
            file_path = img_root_path + file_folder + "/"
            if os.path.exists(file_path):
                file_name_list = os.listdir(file_path)
                if len(file_name_list) == 0:
                    print(file_path, "dont have image!!!"  )
                    # os.remove(file_path)
                    continue
                img_obj = {
                     "image": [ file_path  + file_name for file_name in file_name_list ],
                     "gt" : gt_index,
                }
                photo_set.append(img_obj)
            else:
                print(file_path, "not exists!"  )
##
##
        self.photo_set = photo_set
        self.total_number = len(self.photo_set)
        print("total load data:", self.total_number)
    
    def read_image(self, path, mode = 1): # 默认为彩色图
        img = cv2.imread(path, 0)
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    def set_gan(self, method_list = None):
        
        self.datagan = None
        self.datagan_gt = None
       
        if method_list is None:
            self.datagan = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_shape),
                transforms.ToTensor(),      
                transforms.Normalize(0,1),  
                # transforms.ToPILImage(),
                # transforms.Resize(self.image_shape),
                # transforms.RandomCrop(size = self.image_shape, padding=(10, 20)),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(10),
                # transforms.ToTensor(),      
                # transforms.Normalize(0,1),
            ])
        else:
            self.datagan = transforms.Compose(method_list)
       
        method_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_shape),
            transforms.ToTensor(),      
            transforms.Normalize(0,1),     
        ]
        self.datagan_val = transforms.Compose(method_list)
    
    def __len__(self):
        return len(self.photo_set_one_hot)

    def __getitem__(self, index):
        """
        获取对应index的图像,并视情况进行数据增强
        """
        if index >= self.total_number:
            raise StopIteration
        try:
            re_index = index
            image_gt = self.photo_set_one_hot[re_index]['gt_oh']
            image_src_path_list = self.photo_set_one_hot[re_index]['image']
            img_list = []
            for image_src_path in image_src_path_list:
                
                imgOrigin = self.read_image(image_src_path)
                w,h,c = imgOrigin.shape 
                if w<384:
                    imgOrigin = cv2.resize( imgOrigin, (384, 384) )
                    w,h,c = imgOrigin.shape

                img = imgOrigin[:384,:384,:]
                assert img.shape == (384,384,3), f"{image_src_path}, {imgOrigin.shape}"
                img = self.datagan(img)
          
                img_list.append(img)
                # print(image_src_path, img.shape)
                if h == 768:
                    img = imgOrigin[:384,384:,:]
                    assert img.shape == (384,384,3), f"h, {image_src_path}, {img.shape}"
                    img = self.datagan(img)
                    
                    img_list.append(img)

            image_gt = [ torch.from_numpy(gt) for gt in image_gt ]

            return img_list, image_gt
        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)
            print(index)
            print(traceback.print_exc())

def getFileList(dir_path, file_list, ext = None):
    """
    递归遍历dir path下所有 获取文件夹及其子文件夹中文件 带ext的文件列表
    param dir_path: 文件夹根目录
    param file_list: 文件list地址
    param ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir_path
    if os.path.isfile(dir_path):
        if ext is None:
            file_list.append(dir_path)
        else:
            if ext in dir_path[-3:]:
                file_list.append(dir_path)
    
    elif os.path.isdir(dir_path):
        for s in os.listdir(dir_path):
            newDir=os.path.join(dir_path,s)
            getFileList(newDir, file_list, ext)
 
    return file_list

class DiffDataLoader():
    def __init__(self, dataset, batch_size, drop_last = True, shuffle = True) -> None:
        self.dataloader = dataset
        total = len(self.dataloader)
        self.index = [ i for i in range(total)]
        if shuffle:
            np.random.shuffle(self.index)
        self.len = total//batch_size
        self.batch_size = batch_size
        print("totalbatch:", self.len)
    def __len__(self):
        return self.len
    
    def __iter__(self):
        for index in range(self.len):
            yield self.__getitem__(index)
    
    def __getitem__(self, index):
        if index > self.len:
            raise StopIteration
        
        batch_index = self.index[ index*self.batch_size: (index+1)*self.batch_size]
        img_list = []
        gt_list = [[] for i in range(14)]
        for ind in batch_index:
            img, gt = self.dataloader[ind]
            # print(len(img), img[0].shape)
            img_list.append(img)
            for j in range(len(gt)):
                gt_list[j].append(torch.Tensor(gt[j]))

        for j in range(len(gt_list)):
            gt_list[j] = torch.stack(gt_list[j], 0)
            # print(gt_list[j].shape)

        return img_list, gt_list


class DataloadTest(Dataset):
    def __init__(self, file_path = "") -> None:
        super().__init__()
       
        self.file_path_len = len(file_path)
        file_list = []
        file_list = getFileList(file_path, file_list, "jpg")
        self.load_data(file_list)


    def load_data(self, file_list):
        photo_set = []
        for file_path in file_list:
            file_obj = file_path[self.file_path_len:].replace("\\",'/').split("/")
            folder = file_obj[0]
            name = file_obj[1]
            photo_set.append(
                {
                    "path":file_path,
                    "folder":folder,
                    "name":name,
                }
            )
        self.total = len(photo_set)
        self.photo_set = photo_set

    def __len__(self):
        return self.total



if __name__ == '__main__':
    root_path = r"C:\Users\csz\Desktop\cs\Train"
    image_shape = (384,384)

    a = DataLoad(root_path, image_shape = image_shape)
    a.set_gan()
    print(len(a))
    for i in a:
        img, gt = i
        break

    plt.imshow(img[0].numpy())
    plt.show()
    print("label:", gt)

    # 查询
    name_id = 0 # 0- 13
    class_name = a.data_key[name_id]
    #获取name_id下的这一列的特征
    print(f"class name: {class_name}, info: {a.gt_index2name_dict[name_id][np.argmax(gt[name_id]).item()]}")