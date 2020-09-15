import os
import argparse
import torch
import torch.nn.functional as F
import utils.utils as utils
from data import get_loaders
from torch.autograd import Variable
from models_tmr.vgg11_bn_error import VGG11
#from vgg11_bn_error import VGG11
import torchvision.models as models
from urllib.parse import urlparse
import torch.utils.model_zoo as model_zoo
import re
import os

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--dataset', type=str, default='SubImage')
parser.add_argument('--data', type=str, default='/home/chukexin/hm/Error/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')

args = parser.parse_args()

args.save = 'ModelFile'

def main():
    torch.cuda.set_device(args.gpu)
    pthfile = '/home/xingkouzi/hemeng/error/ModelFile/vgg11_bn.pth'
    model = VGG11()
    model.load_state_dict(torch.load(pthfile))
    #print(net)
    #model = AlexNet_subImagenet()
    #model.load_state_dict(torch.load('./ModelFile/alexnet_imagenet1000.pkl'))
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), 1e-6, momentum=0.9, weight_decay=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    train_loader, reward_loader, valid_loader = get_loaders(args)

    for epoch in range(args.epochs):
        #train_acc = train(train_loader, model, optimizer)

        valid_acc = infer(reward_loader, model)
        print("train_acc:", 0, 'valid_acc', valid_acc)
        if (epoch + 1)%1 == 0:
            torch.save(model.state_dict(), os.path.join(args.save, 'vgg11_1000_tmr.pkl'))

def train(train_loader, model, optimizer):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    total_top5 = utils.AvgrageMeter()

    for step, (data, target) in enumerate(train_loader):
        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()
        '''
        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)
        '''
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(),n)
        total_top5.update(prec5.item(), n)
        print("train_acc:", total_top5.avg)
    return total_top1.avg

def infer(valid_loader, model):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    total_top5 = utils.AvgrageMeter()
    model.eval()

    for step, (data, target) in enumerate(valid_loader):
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        logits = model(data)
        loss = F.cross_entropy(logits, target)
        '''
        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)
        '''
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(),n)
        total_top5.update(prec5.item(), n)
        print('val_acc:',total_top5.avg)
        
    return total_top5.avg

if __name__ == '__main__':
    main()
