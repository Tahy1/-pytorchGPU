import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision

trainData = random.random([10000,3,50,50])
trainData.dtype = 'float32'
trainLabel = random.randint(0,2,100)

trainning = []
for i in range(len(trainLabel)):
    trainning.append([trainData[i], trainLabel[i]])
trainLoader = torch.utils.data.DataLoader(
        trainning,
        batch_size=16,
        shuffle=True,
        pin_memory=True)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3_64 = nn.Conv2d(3,64,3,padding=1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64,2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.relu(self.conv3_64(x))
        x = self.GAP(x)
        x = x.view(x.size()[0], -1)
        out = self.fc1(x)
        return out
net = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)
net.train()
cudnn.benchmark = True
for i, (input, target) in enumerate(trainLoader):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    output = net(input)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())