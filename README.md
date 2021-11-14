# CIFAR10分类

本仓库介绍使用卷积神经网络对CIFAR10数据集进行图像分类，点击[CIFAR](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)可以下载数据集。

## Introduction

使用卷积神经网络对CIFAR10做图像分类，调整超参数观察模型的效果，对实验结果做可视化展示。CIFAR-10是一个用于识别普适物体的小型数据集，它包含了10个类别的RGB彩色图片。

> 该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

![mark](http://markdownsave.oss-cn-beijing.aliyuncs.com/markdown/20191223/194433582.png)

## Installation

### Requirements

- Python=3.8
- Numpy
-  torch==1.7.0
- torchaudio==0.7.0a0+ac17b
- torchvision==0.8.0
- tqdm==4.62.3

### Pytorch

官方链接：

```python
git clone --recursive https://github.com/pytorch/pytorch
```

### Build project

```python
python ImageClassification.py
```



## How to build a CNN 

### 1. Data processing

下载完以后放在data文件夹里面，也可以用自己制作的数据集。

数据预处理的流程：

- 定义transforms，transforms.Compose()可以传入一个列表，该列表是数据增强的类实例，transforms.Compose()能够将这些数据增强的类组合，减少代码量。
- 使用torch.datasets.CIFAR10加载数据集，并传入transform做数据增强

```python
#数据预处理
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=False)
```

### 2. build network

构建一个vgg的网络模型，vgg有好几个版本，这里我创建的是vgg16，这个网络用来提取图片的特征，然后把提取到的特征和连接到10个神经元，也就是分10类

```python
self.classifier = nn.Linear(512, 10)
```

完整的构建网络的代码，用一个类来写。

```
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    #vgg_name = 'VGG16'
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)#输出10类别

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
```

### 3. train

先实例化前面我们创建的模型，然后开出你的gpu或者cpu

后面optimizer是定义一个优化器，scheduler可以动态调整学习率，criterion是定义损失函数，用的是交叉熵损失

```
model = VGG('VGG16').to(torch.device('cuda'))
device = torch.device('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
criterion = nn.CrossEntropyLoss().to(device)
```

训练完了会保存一个模型文件，测试的时候就可以加载这个模型文件，就是代码里的这句

```
torch.save(model, 'model.pkl')
```

完整的训练代码

```python
def train():
    train_los = []
    test_los = []
    epochs = 100
    for epoch in range(1, epochs + 1):
        scheduler.step(epoch)
        print("\n===> epoch: %d/100" % epoch)
        model.train()
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        train_result,_ = train_loss, train_correct / total

        train_los.append(train_result)
        torch.save(model, 'model.pkl')
        print("test:")

        model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
        test_result =  test_loss, test_correct / total
        test_los.append(test_result)
        #画出损失曲线
        plt.figure(1)
        plt.subplot(211)
        plt.plot(train_los)
        plt.subplot(212)
        plt.plot(test_los)
        plt.show()

```

### 4. Test

```python
def test(min_num_image,max_num_image):
    to_pil_image = transforms.ToPILImage()
    cnt = 0
    for image,label in test_loader:
      if cnt<min_num_image or cnt>max_num_image:
            break
      print("label",label)
      out = (model(image.to(device)))
      prediction = torch.max(out, 1)
      print("prediction",prediction[1])
      img = to_pil_image(image[0])
      img.show()
      plt.imshow(img)
      plt.show()
      cnt+=1
```



## 5. Visualize

使用tensorboard查看结果

```bash
tensorboard --logdir=logs
```

![image-20211018112924526](images/image-20211018112924526.png)



## Experiments

### 1. adjust learning

![image-20211018125032836](images/image-20211018125032836.png)

![image-20211018125345312](images/image-20211018125345312.png)



<img src="images/image-20211018155931326.png" alt="image-20211018155931326" style="zoom:50%;" />

实验结果如图所示，可以看到，初始学习率过大，会导致训练收敛比较慢，严重的甚至会导致无法收敛。



### 2.
