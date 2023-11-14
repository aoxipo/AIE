# AIE
AIE is a code for https://tianchi.aliyun.com/competition/entrance/532160?spm=a2c22.12281957.0.0.4c885d9biNyivj



# dataloader

标签关系见test内的文件,使用方法如下，以及标签对应描述信息如  Tag 查询

```
from dataloader import DataLoad

root_path = r"D:\dataset\eye\Train" 
image_shape = (384,384)

a = DataLoad(root_path, image_shape = image_shape)
a.set_gan()
for i in a:
    img, gt = i
    break

plt.imshow(img[0].numpy())
plt.show()
print("label:", gt)

# 查询
name_id = 0 # 0- 13
class_name = a.data_key[name_id]
print(f"class name: {class_name}, info: {a.gt_index2name_dict[name_id][np.argmax(gt[name_id]).item()]}")
```



# train

```python
logger = Logger(
        file_name = "log.txt",
        file_mode= "w+",#"a+",
        should_flush=True
    )

batch_size = 256
image_size = 256 #224 # 7
root_path = # r"C:\Users\csz\Desktop\cs\Train"
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
print("using ",device)
# trainer.load_parameter( "./save_best/GTU_pvt/best.pkl" )

trainer.train_and_test(100, train_loader, validate_loader)
```

