import sys
sys.path.append('F:/AIE-main/')
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


class Train():
    def __init__(self, in_channles, image_size = 320, name = 'dense', method_type = 0, 
                 is_show = True, batch_size = 1, device_ = None, split = False):
        self.in_channels = in_channles
        self.batch_size = batch_size
        self.image_size = image_size
        self.name = name
        self.device = device_
        self.method_type = method_type
        self.lr = 0.0001
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.history_score = []
        self.debug = 3
        self.split = split
        self.create(is_show)

    def create(self, is_show):
        batch_size = self.batch_size
        if self.device is not None:
            device = self.device
        if (self.method_type == 0):
            from model.model import Unet as Model
            self.model = Model()
            print(f"build {self.model.__class__.__name__} model")
        if (self.method_type == 1):
            from model.model import DUAL as Model
            self.model = Model()
            print(f"build {self.model.__class__.__name__} model")
        else:
            raise NotImplementedError
        
        self.cost = torch.nn.MSELoss()
        self.downsample = torch.nn.MaxPool2d(2)
        if (use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
            self.downsample = self.downsample.to(device)
            
        if (is_show):
            summary(self.model, (self.in_channels, self.image_size * 2, self.image_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))


    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_loss = 1000000
        es = 0
        
        self.save_parameter()
        for epoch in range(n_epochs):
            sc = False
            start_time = datetime.datetime.now()
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-" * 10)

            epoch_train_loss = self.train(data_loader_train)
            if epoch % 10 == 0:
                sc = True
            epoch_test_loss = self.test(data_loader_test, sc)

            self.history_acc.append(0)
            self.history_loss.append(epoch_train_loss)
            self.history_test_acc.append(0)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Train Loss is:{:.4f}\nTest Loss is:{:.4f}\ncost time:{:.4f} min, ETA:{:.4f}".format(
                    epoch_train_loss,
                    epoch_test_loss,
                    (datetime.datetime.now() - start_time).seconds / 60,
                    (n_epochs - 1 - epoch) * (datetime.datetime.now() - start_time).seconds / 60,
                )
            )
            if epoch % 10 == 0:
                for image, mask in data_loader_test:
                    mask = mask[0].to(device)
                    output = self.predict_batch(image)
                    break
                display_progress(image[0], mask[0][0].unsqueeze(0), output['mask'][0], edge = output['edge'][0], current_epoch = epoch, save = True, save_path = "./save/" + self.name + "/")
            if (epoch <= 4):
                continue

            if (epoch_test_loss < best_loss):
                best_loss = epoch_test_loss
                es = 0
                self.save_parameter("./save_best/", "best")
            else:
                es += 1
                print("Counter {} of 10".format(es))
                if es > 10:
                    print("Early stopping with best_loss: ", best_loss, "and val_acc for this epoch: ", epoch_test_loss,
                            "...")
                    break
        self.save_history()
        self.save_parameter()

    def test(self, data_loader_test, need_score = False):
        self.model.eval()
        running_loss = 0
        test_index = 0
        iou = []
        dice = []
        pr = []
        nr = []
        with torch.no_grad():
            for data in data_loader_test:
                X_test, y_test = data

                y_gt = y_test
                X_test, y_gt = Variable(X_test).float()
                if (use_gpu):
                    X_test = X_test.to(device)
                    y_gt = y_gt.to(device)
                
                outputs = self.model(X_test)
                # TODO 
                loss1 = self.cost(outputs["mask"], y_gt)
                loss_dice_1 = diceCoeffv2( outputs["mask"], y_gt )
                
                loss = 0.4 * loss_dice_1 + 0.3 * loss1
                running_loss += loss.data.item()
                test_index += 1
                if need_score:
                    gt = y_gt.cpu().numpy()
                    mask = outputs["mask"].detach().cpu().numpy()
                    gt = np.asarray(gt, np.float32)
                    gt /= (gt.max() + 1e-8)

                    mask = np.asarray(mask, np.float32)
                    mask /= (mask.max() + 1e-8)
                    iou.append(mean_iou_np(gt, mask))
                    dice.append(mean_dice_np(gt, mask))
                    pr.append(positive_recall(gt, mask))
                    nr.append(negative_recall(gt, mask))
        # TODO 
        if need_score:
            log_str = "IOU:{:.4f}, DICE:{:.4f}, PR:{:.4f},NR:{:.4f}".format(
                np.mean( iou ), np.mean( dice ), np.mean( pr ),np.mean( nr ),  
            )
            self.history_score.append(log_str)
            print( log_str )

        epoch_loss = running_loss / (test_index + 1)
        return epoch_loss

    def train(self, data_loader_train):
        self.model.train()
        train_index = 0
        running_loss = 0.0

        for data in data_loader_train:
            X_train, y_train = data
            if self.debug:
                self.debug -= 1
                print("X_train :{}".format(X_train.shape,))
                print("Y:", [ i.shape for i in y_train ])
            y_gt = y_train
     
            X_train, y_gt = Variable(X_train).float(), y_gt
            if (use_gpu):
                X_train = X_train.to(device)
                y_gt = y_gt.to(device)

            # print("训练中 train {}".format(X_train.shape))
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train)
            # TODO    Impression_res, HyperF_Type_res, HyperF_Area_res, HyperF_Fovea_res, HyperF_ExtraFovea_res, 
            # HyperF_Y_res, HypoF_Type_res, HypoF_Area_res, HypoF_Fovea_res, 
            # HypoF_ExtraFovea_res, HypoF_Y_res, CNV_res, Vascular_abnormality_res, 
            # Pattern_res 
            loss1 = self.cost(outputs[0], y_gt[0])
            loss_dice_1 = diceCoeffv2( outputs[0], y_gt[0] )
        
            loss2 = self.cost(outputs[1], y_gt[1])
            loss_dice_2 = diceCoeffv2( outputs[1], y_gt[1] )
            
            loss3 = self.cost(outputs[2], y_gt[2])
            loss_dice_3 = diceCoeffv2( outputs[2], y_gt[2] )
            
            loss4 = self.cost(outputs[3], y_gt[3])
            loss_dice_4 = diceCoeffv2( outputs[3], y_gt[3] )
            
            loss5 = self.cost(outputs[4], y_gt[4])
            loss_dice_5 = diceCoeffv2( outputs[4], y_gt[4] )
            
            loss6 = self.cost(outputs[5], y_gt[5])
            loss_dice_6 = diceCoeffv2( outputs[5], y_gt[5] )
            
            loss7 = self.cost(outputs[6], y_gt[6])
            loss_dice_7 = diceCoeffv2( outputs[6], y_gt[6] )
            
            loss8 = self.cost(outputs[7], y_gt[7])
            loss_dice_8 = diceCoeffv2( outputs[7], y_gt[7] )
            
            loss9 = self.cost(outputs[8], y_gt[8])
            loss_dice_9 = diceCoeffv2( outputs[8], y_gt[8] )
            
            loss10 = self.cost(outputs[9], y_gt[9])
            loss_dice_10 = diceCoeffv2( outputs[9], y_gt[9] )
            
            loss11 = self.cost(outputs[10], y_gt[10])
            loss_dice_11 = diceCoeffv2( outputs[10], y_gt[10] )
        
            loss12 = self.cost(outputs[11], y_gt[11])
            loss_dice_12 = diceCoeffv2( outputs[11], y_gt[11] )
        
            loss13 = self.cost(outputs[12], y_gt[12])
            loss_dice_13 = diceCoeffv2( outputs[12], y_gt[12] )
            
            loss14 = self.cost(outputs[13], y_gt[13])
            loss_dice_14 = diceCoeffv2( outputs[13], y_gt[13] )
            
            loss = loss1 + loss_dice_1 +loss2 + loss_dice_2 +loss3 + loss_dice_3 + loss4 + loss_dice_4 + loss5 + loss_dice_5 + loss6 + loss_dice_6 + loss7 + loss_dice_7 + loss8 + loss_dice_8 + loss9 + loss_dice_9 + loss10 + loss_dice_10 
            loss += loss11 + loss_dice_11 +loss12 + loss_dice_12 +loss13 + loss_dice_13 
       
            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()
            train_index += 1

        epoch_train_loss = running_loss / train_index
        return epoch_train_loss

    def predict_batch(self, image):
        if (type(image) == np.ndarray):
            image = torch.from_numpy(image)
        if (len(image.size()) == 3):
            image.unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            image = Variable(image).float()
            if (use_gpu):
                image = image.to(device)
            # print(image.shape)
            output = self.model(image)
        return output

    def save_history(self, file_path='F:/AIE-main/save/'):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "loss_test_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close()
        fo = open(file_path + "test_score.txt", "w+")
        fo.write(str(self.history_score))
        fo.close()

    def save_parameter(self, file_path='F:/AIE-main/save/', name=None):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if name == None:
            file_path = file_path + "model_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(
                "-", "_").replace(".", "_") + ".pkl"
        else:
            file_path = file_path + name + ".pkl"
        torch.save(obj=self.model.state_dict(), f=file_path)

    def load_parameter(self, file_path='./save/'):
        print(f"load:{file_path}")
        self.model.load_state_dict(torch.load(file_path, map_location = device))


device = "cuda:0"
if __name__ == "__main__":

    batch_size = 12
    image_size = 384 #224
    root_path = r"f:\dataset\Train" 
    All_dataloader = DataLoad(
        root_path, 
        image_shape =  (image_size, image_size), #(240, 480), # (320, 640), #(256,256), #(320, 640),
        data_aug = 2,
        )
    
    train_size = int(len(All_dataloader) * 0.8)
    print("size :", train_size)
 
    train_dataset, validate_dataset = torch.utils.data.random_split(All_dataloader
                                                                    , [train_size, len(All_dataloader) - train_size])
    print("train size: {} test size: {} , ".format(len(train_dataset), len( validate_dataset )))
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    validate_loader = DataLoader(
        dataset = validate_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
    )
    method_dict = {
        0: "Unet",
        1: "DUAL",
    }

    trainer = Train( 
        1, image_size,
        name = "aie",
        method_type = 1,
        batch_size = batch_size,
        device_ = "cuda:0",
        is_show = False,
    )
    print(device)
    # trainer.load_parameter( "./save_best/GTU_pvt/best.pkl" )

    trainer.train_and_test(100, train_loader, validate_loader)


