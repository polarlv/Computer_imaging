# 采用我们训练的光场再度训练图像重建模型

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from dataset import MY_MNIST_Train
from module2 import UNet
import matplotlib.pyplot as plt

train_dataset = MY_MNIST_Train(root1='./Pt/STL10_Val_GY_20000-128-128.pt',
                               root2='./Pt/STL_Val_Bucket-20000-1024.pt')

print('done')
torch.cuda.empty_cache()

batch_size = 10
numpattern = 1024  # 是参数pattern的数量
px = 128
pattern = torch.load('./Pt/weights_115.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=3e-4)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 135, 170, 200],
                                                   gamma=0.1)  # 学习率更新策略

model = model.to(device)
criterion = criterion.to(device)

start_epoch = 0
Resume = False
# 恢复上一次训练的网络
if Resume:
    checkpoint_path = "./model_weight_resume/checkpoint_1500.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])  # 加载恢复模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载恢复优化器参数
    start_epoch = checkpoint['epoch']  # 加载恢复上次网络终止是的epoch
    lr_schedule.load_state_dict(checkpoint['lr_schedule'])


# 训练
for epoch in range(start_epoch+1, 231):

    print("-----------------第{}轮训练开始---------------------".format(epoch))
    model.train()  # 作用是启用batch normalization和drop out
    for batch_idx, (data, target) in enumerate(train_loader):  # batch_idx代表的是什么？
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()  # 把梯度置零

        data = data.to(device)
        target = target.to(device)
        data = data.to(torch.float32)

        DGI, output = model(data, pattern, batch_size, numpattern, px)

        output = output.to(torch.float32)  # 将数据类型改变
        target = target.to(torch.float32)

        loss = criterion(output, target)  # 计算损失
        loss.backward()
        optimizer.step()  # 在 batch_idx 层面更新优化器参数

        if batch_idx % 200 == 0:
            print('训练次数: {}，[{}/{} ({:.0f}%)]\t Loss: {}'.format(
                epoch, batch_idx * len(data.cuda()), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    lr_schedule.step()  # 在epoch层面更新学习率

    if epoch % 10 == 0:
        torch.save(model, "./model_weight/STL_Val1/module_{}.pth".format(epoch))  # 每十轮保留一次参数
        print("第{}轮数据已保存".format(epoch))
        print("第{}轮的学习率是：".format(epoch), optimizer.state_dict()['param_groups'][0]['lr'])

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_schedule.state_dict()
        }
        torch.save(checkpoint, './model_weight_resume/checkpoint_%s.pth' % str(epoch))
        print("第{}轮用于恢复的数据已保存".format(epoch))

        target = target.cpu().detach().numpy()
        plt.subplot(152)
        plt.imshow(target[10][0])
        plt.title('target img')
        plt.axis('off')

        TDGI = DGI.cpu().detach().numpy()
        print('DGI:', TDGI[10][0])
        plt.subplot(153)
        plt.imshow(TDGI[10][0])
        plt.title('out DGI')
        plt.axis('off')

        model_out = output.cpu().detach().numpy()
        plt.subplot(154)
        plt.imshow(model_out[10][0])
        plt.title('model out')
        plt.axis('off')

        plt.savefig('learn_result/train_img/{}.jpg'.format(epoch))
        plt.show(block=False)
        plt.pause(2)
        # 清空图像
        plt.close()
