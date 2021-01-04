import torch
import torchvision
from torchvision import datasets,transforms
import torchvision.transforms#重要，，如果不导入显示transforms未定义
from torch.autograd import Variable
import numpy as np
import cv2

#数据转化预处理
transform = transforms.Compose([transforms.ToTensor(),#变为tensor数据类型
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),#对RGB三通道转换
                                transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])])#数据标准化装换，标准差法，mean使用原始数据的均值，std标准差

#数据集的下载，直接在datasets后加上你要下载的数据集的名称
data_train = datasets.MNIST(root = './data/',#下载存放的目录，这里是根目录下的data文件夹，如果没有预先建立，则在下载是自行建立
                            transform = transform,
                            train = True, #train = True表示下载训练数据，False表示下载测试数据
                            download = True)
data_test = datasets.MNIST(root = './data/',
                           transform = transform,
                           train = False)
#加载数据，batch_size确定每个包的大小，这里64则显示64张图，shuffle=true表示随机打乱顺序并进行打包处理
train_loader = torch.utils.data.DataLoader(dataset = data_train, batch_size =64, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = data_test, batch_size =64, shuffle = True)

#迭代获取一个批次的数据及其标签
images, labels = next(iter(train_loader))

#一个批次数据的加载,4维，分别是batch_size,channel,height,weight
#经过make_grid 之后图片维度变为（channel,height,weight)，一个批次的数据都整合到了一起
img = torchvision.utils.make_grid(images)
#numpy转置
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]

img = img * std + mean

#print([labels[i].item() for i in range(64)])

#cv2.imshow('mnist_train',img)
#key_pressed = cv2.waitKey(0)

class Model(torch.nn.Module):

    def __init__(self):  
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            #卷积层，参数包括输入通道数，输出通道数，卷积核大小，卷积核移动步长和padding值
            #输入通道变为3
            torch.nn.Conv2d(3, 64, kernel_size = 3, stride = 3, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),#调整了一下通道数，可以实现但为什么那么慢嘞？
            torch.nn.ReLU(),
            #最大池化层，参数包括池化窗口大小，池化窗口移动步长和padding值
            torch.nn.MaxPool2d(stride = 2, kernel_size = 2)
        )
        #全连接层
        self.dense = torch.nn.Sequential(
        
            torch.nn.Linear(5*5*64, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 0.5),#防止发生过拟合，以一定的随机概率将部分参数归零
            torch.nn.Linear(1024,10)
        )
    #向前传播
    def forward(self,x):  
        x = self.conv1(x)#卷积处理
        x = x.view(-1, 5 *5 *64) #对参数进行扁平化处理，否则输入参数和输出参数的维度不匹配
        x = self.dense(x)
        return x

model = Model()
#用交叉熵来计算损失值
cost = torch.nn.CrossEntropyLoss()
#用Adam优化参数
optimizer = torch.optim.Adam(model.parameters())
#print(model)


epoch_n = 5

#模型训练和参数优化
for epoch in range(epoch_n):
    #初始化损失和正确率
    running_loss = 0.0   
    running_correct = 0
    print("Epoch {}/{}".format(epoch, epoch_n))
    print("-"*10)

    for data in train_loader:  
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)
        outputs = model(x_train)
        _,pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs,y_train)
        loss.backward() 
        optimizer.step()
        running_loss += loss.data.item()
        running_correct += torch.sum(pred == y_train.data)   

    testing_correct = 0

    for data in test_loader:  
        x_test, y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)
        outputs = model(x_test)
        _,pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    #print('Loss is: {:.4f}, Train Accuracy is: {:.4f}, Test Accuracy is: {:.4f}'.format(running_loss/len(data_train), 100 * running_correct / len(data_train), 100*testing_correct / len(data_test))) 


test_loader = torch.utils.data.DataLoader(dataset = data_test, batch_size = 4, shuffle = True)
x_test, y_test = next(iter(test_loader))
inputs = Variable(x_test)
pred = model(inputs)
_,pred = torch.max(pred, 1)
print("Predict Label is: ", [i for i in pred.data])
print('Real Label is: ', [i for i in y_test])

img = torchvision.utils.make_grid(x_test)
img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]

img = img * std + mean

cv2.imshow('mnist_train',img)
key_pressed = cv2.waitKey(0)

#可以实现预测，但是为什么花费那么长的时间呢