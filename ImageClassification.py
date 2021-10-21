#!/usr/bin/env python
# coding: utf-8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torchvision
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=False)


cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = VGG('VGG16').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4], gamma=0.5)
criterion = nn.CrossEntropyLoss().to(device)


writer = SummaryWriter('logs')
accuracy = 0
acc = []
los = []
epochs = 10
bar = tqdm(range(1, epochs + 1), desc='epochs')
for epoch in bar:
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
    print(scheduler.get_last_lr())
    scheduler.step()
    train_result,train_acc = train_loss, train_correct / total
    writer.add_scalar('train_loss', train_result, epoch)
    writer.add_scalar('trian_accuracy', train_acc, epoch)
    acc.append(train_acc)
    los.append(train_result)
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

    test_result = test_loss, test_correct / total

    print("train loss=%.5f" % train_loss)
    print("train accuracy=%.5f" % train_acc)
    print("test loss=%.5f" % test_result[0])
    print("test accuracy=%.5f" % test_result[1])
    accuracy = max(accuracy, test_result[1])
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_accuracy', accuracy, epoch)


# import matplotlib.pyplot as plt
# plt.plot(acc)
# plt.plot(los)



# cnt = 0
# to_pil_image = transforms.ToPILImage()
# for image,label in test_loader:
#   if cnt>=3:      # 只显示3张图片
#         break
#   print(label)
#   out = (model(image.to(device)))
#   prediction = torch.max(out, 1)
#   print(prediction[1])
#   img = to_pil_image(image[0])
#   img.show()
#   plt.imshow(img)
#   plt.show()
#   cnt = cnt +1




#
# simple_transform = transforms.Compose([transforms.Resize((224,224))
#                                        ,transforms.ToTensor()
#                                        ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                       ])
# train = ImageFolder('dogsandcats/train/',simple_transform)
# valid = ImageFolder('dogsandcats/valid/',simple_transform)
#
#
#
#
# print(train.class_to_idx)
# print(train.classes)


#
# train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=3,shuffle=True)
# valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=32,num_workers=3,shuffle=True)


# In[ ]:


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(56180, 500)
#         self.fc2 = nn.Linear(500,50)
#         self.fc3 = nn.Linear(50, 2)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         x = F.dropout(x,training=self.training)
#         x = self.fc3(x)
#         return F.log_softmax(x,dim=1)
#
#
# # In[ ]:
#
#
# model = Net()
# if is_cuda:
#     model.cuda()
#
#
# # In[ ]:
#
#
# optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
#
#
# # In[ ]:
#
#
# def fit(epoch,model,data_loader,phase='training',volatile=False):
#     if phase == 'training':
#         model.train()
#     if phase == 'validation':
#         model.eval()
#         volatile=True
#     running_loss = 0.0
#     running_correct = 0
#     for batch_idx , (data,target) in enumerate(data_loader):
#         if is_cuda:
#             data,target = data.cuda(),target.cuda()
#         data , target = Variable(data,volatile),Variable(target)
#         if phase == 'training':
#             optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output,target)
#
#         running_loss += F.nll_loss(output,target,size_average=False).data[0]
#         preds = output.data.max(dim=1,keepdim=True)[1]
#         running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
#         if phase == 'training':
#             loss.backward()
#             optimizer.step()
#
#     loss = running_loss/len(data_loader.dataset)
#     accuracy = 100. * running_correct/len(data_loader.dataset)
#
#     print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
#     return loss,accuracy
#
#
# # In[ ]:
#
#
# train_losses , train_accuracy = [],[]
# val_losses , val_accuracy = [],[]
# for epoch in range(1,20):
#     epoch_loss, epoch_accuracy = fit(epoch,model,train_data_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,model,valid_data_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)
#
#
# # In[ ]:
#
#
# plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
# plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
# plt.legend()
#
#
# # In[ ]:
#
#
# plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
# plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
# plt.legend()
#
#
# # ## Transfer learning using VGG
#
# # In[ ]:
#
#
# vgg = models.vgg16(pretrained=True)
# vgg = vgg.cuda()
#
#
# # ### Print VGG
#
# # In[ ]:
#
#
# vgg
#
#
# # ## Freeze layers
#
# # In[ ]:
#
#
# vgg.classifier[6].out_features = 2
# for param in vgg.features.parameters(): param.requires_grad = False
#
#
# # In[ ]:
#
#
# optimizer = optim.SGD(vgg.classifier.parameters(),lr=0.0001,momentum=0.5)
#
#
# # In[ ]:
#
#
# def fit(epoch,model,data_loader,phase='training',volatile=False):
#     if phase == 'training':
#         model.train()
#     if phase == 'validation':
#         model.eval()
#         volatile=True
#     running_loss = 0.0
#     running_correct = 0
#     for batch_idx , (data,target) in enumerate(data_loader):
#         if is_cuda:
#             data,target = data.cuda(),target.cuda()
#         data , target = Variable(data,volatile),Variable(target)
#         if phase == 'training':
#             optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output,target)
#
#         running_loss += F.cross_entropy(output,target,size_average=False).data[0]
#         preds = output.data.max(dim=1,keepdim=True)[1]
#         running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
#         if phase == 'training':
#             loss.backward()
#             optimizer.step()
#
#     loss = running_loss/len(data_loader.dataset)
#     accuracy = 100. * running_correct/len(data_loader.dataset)
#
#     print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
#     return loss,accuracy
#
#
# # In[ ]:
#
#
# train_losses , train_accuracy = [],[]
# val_losses , val_accuracy = [],[]
# for epoch in range(1,10):
#     epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)
#
#
# # ### Adjusting dropout
#
# # In[ ]:
#
#
# for layer in vgg.classifier.children():
#     if(type(layer) == nn.Dropout):
#         layer.p = 0.2
#
#
# # In[ ]:
#
#
# train_losses , train_accuracy = [],[]
# val_losses , val_accuracy = [],[]
# for epoch in range(1,3):
#     epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)
#
#
# # ### Data augmentation
#
# # In[ ]:
#
#
# train_transform = transforms.Compose([transforms.Resize((224,224))
#                                        ,transforms.RandomHorizontalFlip()
#                                        ,transforms.RandomRotation(0.2)
#                                        ,transforms.ToTensor()
#                                        ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                       ])
# train = ImageFolder('dogsandcats/train/',train_transform)
# valid = ImageFolder('dogsandcats/valid/',simple_transform)
#
#
# # In[ ]:
#
#
# train_data_loader = DataLoader(train,batch_size=32,num_workers=3,shuffle=True)
# valid_data_loader = DataLoader(valid,batch_size=32,num_workers=3,shuffle=True)
#
#
# # In[ ]:
#
#
# train_losses , train_accuracy = [],[]
# val_losses , val_accuracy = [],[]
# for epoch in range(1,3):
#     epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
#     val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
#     train_losses.append(epoch_loss)
#     train_accuracy.append(epoch_accuracy)
#     val_losses.append(val_epoch_loss)
#     val_accuracy.append(val_epoch_accuracy)
#
#
# # ## Calculating preconvoluted features
# #
#
# # In[ ]:
#
#
# vgg = models.vgg16(pretrained=True)
# vgg = vgg.cuda()
#
#
# # In[ ]:
#
#
# features = vgg.features
#
#
# # In[ ]:
#
#
# for param in features.parameters(): param.requires_grad = False
#
#
# # In[ ]:
#
#
# train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=3,shuffle=False)
# valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=32,num_workers=3,shuffle=False)
#
#
# # In[ ]:
#
#
# def preconvfeat(dataset,model):
#     conv_features = []
#     labels_list = []
#     for data in dataset:
#         inputs,labels = data
#         if is_cuda:
#             inputs , labels = inputs.cuda(),labels.cuda()
#         inputs , labels = Variable(inputs),Variable(labels)
#         output = model(inputs)
#         conv_features.extend(output.data.cpu().numpy())
#         labels_list.extend(labels.data.cpu().numpy())
#     conv_features = np.concatenate([[feat] for feat in conv_features])
#
#     return (conv_features,labels_list)
#
#
# # In[ ]:
#
#
# conv_feat_train,labels_train = preconvfeat(train_data_loader,features)
#
#
# # In[ ]:
#
#
# conv_feat_val,labels_val = preconvfeat(valid_data_loader,features)
#
#
# # In[ ]:
#
#
# class My_dataset(Dataset):
#     def __init__(self,feat,labels):
#         self.conv_feat = feat
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.conv_feat)
#
#     def __getitem__(self,idx):
#         return self.conv_feat[idx],self.labels[idx]
#
#
# # In[ ]:
#
#
# train_feat_dataset = My_dataset(conv_feat_train,labels_train)
# val_feat_dataset = My_dataset(conv_feat_val,labels_val)
#
#
# # In[ ]:
#
#
# train_feat_loader = DataLoader(train_feat_dataset,batch_size=64,shuffle=True)
# val_feat_loader = DataLoader(val_feat_dataset,batch_size=64,shuffle=True)
#
#
# # In[ ]:
#
#
# def data_gen(conv_feat,labels,batch_size=64,shuffle=True):
#     labels = np.array(labels)
#     if shuffle:
#         index = np.random.permutation(len(conv_feat))
#         conv_feat = conv_feat[index]
#         labels = labels[index]
#     for idx in range(0,len(conv_feat),batch_size):
#         yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])
#
#
# # In[ ]:
#
#
# train_batches = data_gen(conv_feat_train,labels_train)
# val_batches = data_gen(conv_feat_val,labels_val)
#
#
# # In[ ]:
#
#
# optimizer = optim.SGD(vgg.classifier.parameters(),lr=0.0001,momentum=0.5)
#
#
# # In[ ]:
#
#
# def fit_numpy(epoch,model,data_loader,phase='training',volatile=False):
#     if phase == 'training':
#         model.train()
#     if phase == 'validation':
#         model.eval()
#         volatile=True
#     running_loss = 0.0
#     running_correct = 0
#     for batch_idx , (data,target) in enumerate(data_loader):
#         if is_cuda:
#             data,target = data.cuda(),target.cuda()
#         data , target = Variable(data,volatile),Variable(target)
#         if phase == 'training':
#             optimizer.zero_grad()
#         data = data.view(data.size(0), -1)
#         output = model(data)
#         loss = F.cross_entropy(output,target)
#
#         running_loss += F.cross_entropy(output,target,size_average=False).data[0]
#         preds = output.data.max(dim=1,keepdim=True)[1]
#         running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
#         if phase == 'training':
#             loss.backward()
#             optimizer.step()
#
#     loss = running_loss/len(data_loader.dataset)
#     accuracy = 100. * running_correct/len(data_loader.dataset)
#
#     print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
#     return loss,accuracy
#
#
# # In[ ]:
#
#
# get_ipython().run_cell_magic('time', '', "train_losses , train_accuracy = [],[]\nval_losses , val_accuracy = [],[]\nfor epoch in range(1,20):\n    epoch_loss, epoch_accuracy = fit_numpy(epoch,vgg.classifier,train_feat_loader,phase='training')\n    val_epoch_loss , val_epoch_accuracy = fit_numpy(epoch,vgg.classifier,val_feat_loader,phase='validation')\n    train_losses.append(epoch_loss)\n    train_accuracy.append(epoch_accuracy)\n    val_losses.append(val_epoch_loss)\n    val_accuracy.append(val_epoch_accuracy)")
#
#
# # ## Visualizing intermediate CNN layers
#
# # In[ ]:
#
#
# train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=3,shuffle=False)
# img,label = next(iter(train_data_loader))
#
#
# # In[ ]:
#
#
# imshow(img[5])
#
#
# # In[ ]:
#
#
# img = img[5][None]
#
#
# # In[ ]:
#
#
# vgg = models.vgg16(pretrained=True).cuda()
#
#
# # In[ ]:
#
#
# class LayerActivations():
#     features=None
#
#     def __init__(self,model,layer_num):
#         self.hook = model[layer_num].register_forward_hook(self.hook_fn)
#
#     def hook_fn(self,module,input,output):
#         self.features = output.cpu().data.numpy()
#
#     def remove(self):
#         self.hook.remove()
#
#
# conv_out = LayerActivations(vgg.features,0)
#
# o = vgg(Variable(img.cuda()))
#
# conv_out.remove()
#
#
# # In[ ]:
#
#
# act = conv_out.features
#
#
# # In[ ]:
#
#
# fig = plt.figure(figsize=(20,50))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
#     ax.imshow(act[0][i])
#
#
# # In[ ]:
#
#
# vgg
#
#
# # In[ ]:
#
#
# conv_out = LayerActivations(vgg.features,1)
#
# o = vgg(Variable(img.cuda()))
#
# conv_out.remove()
#
# act = conv_out.features
#
# fig = plt.figure(figsize=(20,50))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
#     ax.imshow(act[0][i])
#
#
# # In[ ]:
#
#
# def imshow(inp):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#
#
# # In[ ]:
#
#
# conv_out = LayerActivations(vgg.features,1)
#
# o = vgg(Variable(img.cuda()))
#
# conv_out.remove()
#
# act = conv_out.features
#
#
#
#
#
# fig = plt.figure(figsize=(20,50))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(12,5,i+1,xticks=[],yticks=[])
#     ax.imshow(act[0][i])
#
#
# # ## Visualizing weights
#
# # In[ ]:
#
#
# vgg = models.vgg16(pretrained=True).cuda()
#
#
# # In[ ]:
#
#
# vgg.state_dict().keys()
#
#
# # In[ ]:
#
#
#
# cnn_weights = vgg.state_dict()['features.0.weight'].cpu()
#
#
# # In[ ]:
#
#
# fig = plt.figure(figsize=(30,30))
# fig.subplots_adjust(left=0,right=1,bottom=0,top=0.8,hspace=0,wspace=0.2)
# for i in range(30):
#     ax = fig.add_subplot(12,6,i+1,xticks=[],yticks=[])
#     imshow(cnn_weights[i])
#
#
# # In[ ]:
#
#
# cnn_weights.shape
#
#
# # In[ ]:




