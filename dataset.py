import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MY_MNIST_Train(Dataset):

    def __init__(self, root1, root2, transform=None):
        self.transform = transform
        pre_data = torch.load(root2)  # 网络的输入
        self.targets = torch.load(root1)  # 标签、原图
        # print("原始数据为{}".format(pre_data[0][0][0]))

        self.data = pre_data

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.unsqueeze(0)
        target = target.unsqueeze(0)
        return img, target

    def __len__(self):
        return len(self.data)


class MY_MNIST_Test(Dataset):

    def __init__(self, root, transform=None):
        self.transform = transform
        pre_data = torch.load(root)
        old_data = torch.load(root)
        # print("原始数据为{}".format(pre_data[0][0][0]))
        self.data = pre_data

    def __getitem__(self, index):
        img = self.data[index]
        img = img.unsqueeze(0)
        return img

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    data = MY_MNIST_Train(root1=r'./train-pt/1.pt',
                               root2=r'./train-GI-pt/1.pt')
#     dataloader = DataLoader(data, batch_size=16, shuffle=True,num_workers=0,drop_last=True)
#     for i,(img,label) in  enumerate(dataloader):
#         print(img.shape)
    a, b = data[0]
    print(a.shape)
