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



