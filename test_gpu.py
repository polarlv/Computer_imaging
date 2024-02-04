import time
import torch
import cv2 as cv
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image

batch_size = 1
px = 128
pattern = torch.load('./Pt/weights_115.pt')
pattern = pattern.squeeze(1)
c_pattern = pattern.cpu().detach()
numpattern = pattern.shape[0]

toPIL = transforms.ToPILImage()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
model = torch.load("model_weight/STL_Val4/module_350.pth")
print("模型加载成功！")
model.eval()

print('正式运行开始！')
start_time = time.time()
for s in range(10):

    time1 = time.time()
    imgpath = "Test_img/{}.jpg".format(s)  # 图片路径
    img = cv.imread(imgpath, 0)  # 读取图片,0代表灰度
    mi = np.min(img)
    mx = np.max(img)
    img = (img - mi) / (mx - mi)
    obt = torch.tensor(img)
    obt = torch.reshape(obt, (1, px, px))
    obt_field = obt * c_pattern
    bucket = obt_field.sum(2).sum(1)
    data = bucket
    # data = torch.load('Pt/1024_sy_learn_bucket.pt')*10000
    time2 = time.time()
    # print('采集一幅图像的时间为：', time2 - time1)

    with torch.no_grad():

        data = data.to(torch.float32)
        data = data.to(device)
        starter.record()
        DGI, output = model(data, pattern, batch_size, numpattern, px)
        ender.record()
        torch.cuda.synchronize()
        DGI_img = DGI[0][0].cpu().numpy()
        img = output[0][0].cpu().numpy()  # tensor[1,1,128,128]
        # cv.imwrite('./test_result/GI_img/Sy_GI{}.bmp'.format(s), DGI_img*255)
        # cv.imwrite('./test_result/Sy/sy{}.bmp'.format(s), img * 255)
        print('重建一幅图像的时间为：', starter.elapsed_time(ender)/1000)

        cv.imshow('GI_image', img)
        cv.waitKey(1)


end_time = time.time()
print("预测图片累计耗时{}s".format(end_time - start_time))
