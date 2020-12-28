import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from torchsummary import summary

from torch.optim.lr_scheduler import ReduceLROnPlateau

def summary_trainable(learner):
    """
    Return the model architechture with the number of trainable parameters per layer
    """
    result = []
    total_params_element = 0
    def check_trainable(module):
        nonlocal total_params_element
        if len(list(module.children())) == 0:
            num_param = 0
            num_trainable_param = 0
            num_param_numel = 0
            for parameter in module.parameters():
                num_param += 1
                if parameter.requires_grad:
                    num_param_numel += parameter.numel()
                    total_params_element += parameter.numel()
                    num_trainable_param += 1

            result.append({'module': module, 'num_param': num_param , 'num_trainable_param' : num_trainable_param, 'num_param_numel': num_param_numel})
    learner.apply(check_trainable)
    print("{: <85} {: <17} {: <20} {: <40}".format('Module Name', 'Total Parameters', 'Trainable Parameters', '# Elements in Trainable Parametrs'))
    for row in result:
        print("{: <85} {: <17} {: <20} {: <40,}".format(row['module'].__str__(), row['num_param'], row['num_trainable_param'], row['num_param_numel']))
        print('Total number of parameters elements {:,}'.format(total_params_element))

# Training settings

parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='./data', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
from data import data_transforms_pretrained

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms_pretrained['train']),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms_pretrained['val']),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Pretrained_model, Efficientnet
model = Pretrained_model("resnext101", n_frozen_layers=8)
#print(summary_trainable(model))

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

#set a learning scheduler that reduces the LR if the validation accuracy stops increasing
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.8, patience=5)

def train(epoch,hist_loss, hist_acc):
    model.train()
    train_loss = 0.
    correct = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()


        #augment train loss epoch
        train_loss += loss.data.item()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    train_loss /= len(train_loader.dataset)
    hist_loss.append(train_loss)
    hist_acc.append(float(correct) / len(train_loader.dataset))

def validation(hist_loss, hist_acc):
    model.eval()
    validation_loss = 0.
    correct = 0.
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    validation_acc = correct / len(val_loader.dataset)
    hist_loss.append(validation_loss)
    hist_acc.append(validation_acc)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * validation_acc))
    return validation_acc


if __name__ == '__main__':
    hist_val_loss = []
    hist_train_loss = []

    hist_val_acc = []
    hist_train_acc = []

    for epoch in range(1, args.epochs + 1):
        train(epoch, hist_train_loss, hist_train_acc)
        val_acc = validation(hist_val_loss, hist_val_acc)

        scheduler.step(val_acc)

        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
    
    epochs = [epoch for epoch in range(1,args.epochs+1)]

    #save training/validation loss plots
    plt.figure()
    plt.plot(epochs, hist_train_loss,label="train_loss")
    plt.plot(epochs, hist_val_loss, label="val_loss")

    lines = plt.gca().get_lines()
    include = [0,1]
    legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)

    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig(args.experiment + '/loss.jpg')

    #save training/validation accuracy_plots
    plt.figure()
    plt.plot(epochs, hist_train_acc,label="train_acc")
    plt.plot(epochs, hist_val_acc, label="val_acc")

    lines = plt.gca().get_lines()
    include = [0,1]
    legend1 = plt.legend([lines[i] for i in include],[lines[i].get_label() for i in include], loc=1)

    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.savefig(args.experiment + '/accuracy.jpg')
