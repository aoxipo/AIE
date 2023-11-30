import sys
sys.path.append('C:/Users/csz/Desktop/cs/AIE')
from dataloader import DataLoad
from dataloader import DiffDataLoader as DataLoader
# from torch.utils.data import DataLoader
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
DATA_KEY = [ "Impression", "HyperF_Type", "HyperF_Area(DA)", "HyperF_Fovea", "HyperF_ExtraFovea", "HyperF_Y", 
      "HypoF_Type" ,"HypoF_Area(DA)","HypoF_Fovea", "HypoF_ExtraFovea"
    ,"HypoF_Y","CNV","Vascular abnormality (DR)","Pattern"]

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
        self.cross_coss = torch.nn.CrossEntropyLoss()
        self.downsample = torch.nn.MaxPool2d(2)
        if (use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
            self.cross_coss = self.cross_coss.to(device)
            self.downsample = self.downsample.to(device)
            
        if (is_show):
            summary(self.model, (self.in_channels, self.image_size * 2, self.image_size))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))


    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_loss = 1000000
        es = 0
        epoch_train_loss = 0
        epoch_test_loss = 0
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
            # if epoch % 10 == 0:
            #     for image, mask in data_loader_test:
            #         mask = mask[0].to(device)
            #         output = self.predict_batch(image)
            #         break
            #     display_progress(image[0], mask[0][0].unsqueeze(0), output['mask'][0], edge = output['edge'][0], current_epoch = epoch, save = True, save_path = "./save/" + self.name + "/")
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
        self.save_parameter("./save/", "best")

    def test(self, data_loader_test, need_score = False):
        self.model.eval()
        running_loss = 0
        test_index = 0
        acc = []
        acc_dict = {}
        for i in DATA_KEY:
            acc_dict[i] = []

        with torch.no_grad():
            for data in data_loader_test:

                X_test, y_test = data
                X_test, y_gt = Variable(X_test).float(), y_test
                if (use_gpu):
                    X_test = X_test.to(device)
                    for i in range(len(y_gt)):
                        y_gt[i] = y_gt[i].to(device)
                
                outputs = self.model(X_test)

                loss = self.cost(outputs[0], y_gt[0].float())
                loss_cross = self.cross_coss(  outputs[0], torch.argmax(y_gt[0], 1).long()  )
                for i in range(1,14):
                    loss += self.cost(outputs[i], y_gt[i].float())
                    loss_cross += self.cross_coss(  outputs[i], torch.argmax(y_gt[i], 1).long()  )
                    
                running_loss += loss.data.item()
                test_index += 1
                if need_score:
                    
                    acc_1 = torch.sum(torch.argmax(outputs[0], 1) == torch.argmax(y_gt[0], 1))/len(y_gt[0])
                    acc_dict[DATA_KEY[0]].append( acc_1.cpu().item() )
                    acc_list = [acc_1]
                    for i in range(1,14):
                        acc_list.append( torch.sum(torch.argmax(outputs[i], 1) == torch.argmax(y_gt[i], 1))/len(y_gt[i]))
                        acc_dict[DATA_KEY[i]].append((torch.sum(torch.argmax(outputs[i], 1) == torch.argmax(y_gt[i], 1)) / len(y_gt[i])).cpu().item())


                    acc.append( acc_list )


        if need_score:
            total_acc = []
            for key, value in acc_dict.items():
                total_acc.append( np.mean( value ) )
            log_str = "Total ACC:{:.4f}, ".format(
                np.mean( total_acc )
            )
            for i in range(len(DATA_KEY)):
                log_str += f"{DATA_KEY[i]} ACC:{  np.array(acc_dict[DATA_KEY[i]]).mean():.4f}, "
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
                # print("X_train :{},{} ".format(X_train[0].shape,len(X_train)))
                print("Y:", [ i[0].shape for i in y_train ])
            y_gt = y_train
     
            X_train, y_gt = X_train, y_gt
            if (use_gpu):
                for i in range(len(X_train)): 
                    for j in range(len(X_train[i])):
                        # print((X_train[i][j])
                        X_train[i][j] = Variable(X_train[i][j]).float().to(device)
                print(len(y_gt), len(y_gt[0]))
                for i in range(len(y_gt)):
                    y_gt[i]= y_gt[i].to(device)

            # print("训练中 train {}".format(X_train.shape))
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train) # n  B, C
           
            loss = self.cost(outputs[0], y_gt[0].float())
            # print(outputs[0].shape,  y_gt[0].shape, torch.argmax(y_gt[0], 1).long(), torch.argmax(y_gt[0], 1).long().shape)
            loss_cross = self.cross_coss(  outputs[0], torch.argmax(y_gt[0], 1).long()  )
            for i in range(1,14):
                loss += self.cost(outputs[i], y_gt[i].float())
                loss_cross += self.cross_coss(  outputs[i], torch.argmax(y_gt[i], 1 ).long()  )
            
            loss = loss + loss_cross
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

    def save_history(self, file_path='./save/'):
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

    def save_parameter(self, file_path='./save/', name=None):
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
    logger = Logger(
        file_name = "log.txt",
        file_mode= "w+",#"a+",
        should_flush=True
    )
    batch_size = 12
    image_size = 256 #224 # 7
    root_path = r"D:\dataset\eye\Train" #r"C:\Users\csz\Desktop\cs\Train"
    All_dataloader = DataLoad(
        root_path, 
        image_shape =  (image_size, image_size), #(240, 480), # (320, 640), #(256,256), #(320, 640),
        data_aug = 1,
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
    print("############################", len(train_loader), ":train_dataset:", len(train_dataset))
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

    # for i in train_loader:
    #     print("get data tarin loader #################################", i[0].shape)
    #     break

    trainer = Train( 
        1, image_size,
        name = "aie",
        method_type = 1,
        batch_size = batch_size,
        device_ = "cuda:0",
        is_show = False,
    )
    print("using ",device)
    # trainer.load_parameter( "./save_best/GTU_pvt/best.pkl" )

    trainer.train_and_test(101, train_loader, validate_loader)


