import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.sgd import SGD
import os
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F

model_save_dir = '/data/mnist/models'

# Hyper Parameters 
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.01

logger = Logger('momentum.txt', title='mnist')
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    
net = Net(input_size, hidden_size, num_classes)
net.cuda()   
net.train()    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
# Train the Model
iters = 0
for epoch in range(num_epochs):

    train_loss_log = AverageMeter()
    train_acc_log = AverageMeter()
    val_loss_log = AverageMeter()
    val_acc_log = AverageMeter()    
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28).cuda())
        labels = Variable(labels.cuda())
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        train_loss_log.update(train_loss.data[0], images.size(0))
        train_acc_log.update(prec1[0], images.size(0))
        
        save_name = os.path.join(model_save_dir, str(iters)+'.pth.tar')
        torch.save({'iter': iters,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict()}, 
                    save_name)

        iters+=1
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, train_loss_log.avg, train_acc_log.avg))

    # Test the Model
    net.eval()
    correct = 0
    loss = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28)).cuda()
        labels = Variable(labels).cuda()
        outputs = net(images)
        test_loss = criterion(outputs, labels)
        val_loss_log.update(test_loss.data[0], images.size(0))
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        val_acc_log.update(prec1[0], images.size(0))
        
    logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
    print('Accuracy of the network on the 10000 test images: %.8f %%' % (val_acc_log.avg))
    print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))
    
logger.close()
logger.plot()

