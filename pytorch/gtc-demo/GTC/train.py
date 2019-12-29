import torch

import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import time

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Module
import math
import argparse

import perseus.torch.horovod as hvd


# parameters
def parse_args():
    parser = argparse.ArgumentParser(description='GTC-demo')
    # general
    parser.add_argument('--end-epoch', type=int, default=20, help='training epoch size')
    parser.add_argument('--test-size', type=int, default=50, help='test data size')
    parser.add_argument('--model', type=str, default='resnet50', help='finetune model type')
    parser.add_argument('--num-classes', type=int, default=3, help='num classes')
    parser.add_argument('--pretrain', type=int, default=0, help='1: use pretrain model')
    parser.add_argument('--pretrain-model', type=str, default='', help='pretrain model file')
    parser.add_argument('--decay-epoch', type=int, default=10, help='decay lr each epoch')
    parser.add_argument('--decay-lr', type=int, default=10.0, help='decay lr rate')
    parser.add_argument('--save-model', type=str, default='save_model.pth', help='save model to a file')
    parser.add_argument('--no-freeze-layer', type=str, default='fc', help='no freeze layer list')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize of each GPU')
    parser.add_argument('--prefetch-nums', type=int, default=4, help='num prefetch data of each GPU')
    parser.add_argument('--l2-norm', type=int, default=0, help='use l2 norm')
    parser.add_argument('--dropout', type=int, default=1, help='1: use dropout')
    parser.add_argument('--dropout-rate', type=float, default=0.25, help='dropout rate')
    parser.add_argument('--color-jit', type=int, default=0, help='1: use colorjit to enhance data')
    parser.add_argument('--random-gray', type=int, default=0, help='1: use random-gray to enhance data')
    parser.add_argument('--use-margin', type=int, default=0, help='1: use margin loss')
    parser.add_argument('--margin-m', type=float, default=0.35, help='cosine margin m')
    parser.add_argument('--margin-s', type=float, default=30., help='feature scale s')
    
    args = parser.parse_args()
    return args


def margin(cos, label, m, s):
    #m = 0.35
    #s = 30.
    phi = cos - m 
    label = label.view(-1, 1)
    index = cos.data * 0.0
    index.scatter_(1, label.data.view(-1, 1), 1)
    index = index.byte()
    output = cos * 1.0
    output[index] = phi[index]
    output *= s
    return output

def train(args):
    t0 = time.time()

    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomGrayscale(p=0.5) if args.random_gray else '',
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1) if args.color_jit else '',
            transforms.ColorJitter(brightness=1) if args.color_jit else '', 
            transforms.ColorJitter(contrast=1) if args.color_jit else '',
            transforms.ColorJitter(saturation=0.5) if args.color_jit else '',
            transforms.ColorJitter(hue=0.5) if args.color_jit else '',
            transforms.Resize((224, 224)),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # data loader
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.test_size, args.test_size])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        sampler=train_sampler,
        num_workers=args.prefetch_nums
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.prefetch_nums
    )

    t1 = time.time()

    if args.model == 'alexnet':
        model = models.alexnet(pretrained=True) 
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True) 
        model.fc = torch.nn.Linear(2048, args.num_classes)
    else:
        raise NotImplementedError()

    # freeze layer
    for p in model.parameters():
        p.requires_grad = False 

    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain_model))

    for p in model.fc.parameters():
        p.requires_grad = True
    '''
    for name, child in model.named_children():
       #if name in ['layer3', 'layer4', 'fc']:
       if name in ['layer4', 'fc']:
           print(name + ' is unfrozen')
           for param in child.parameters():
               param.requires_grad = True
       else:
           print(name + ' is frozen')
           for param in child.parameters():
               param.requires_grad = False
    '''         

    # perseus 2: model to cuda
    model = model.cuda() 

    NUM_EPOCHS = args.end_epoch
    BEST_MODEL_PATH = args.save_model 
    best_accuracy = 0.0

    # perseus 3: lr * hvd.size()
    lr = lr * hvd.size()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    
    # perseus 4: wrapper optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # perseus 5: broadcast model parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # perseus 6: broadcast opt state
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    t2 = time.time()
    all_time = 0
    criterion = F.cross_entropy
    model.train()
    final_save_epoch = 0
    for epoch in range(NUM_EPOCHS):
        t3 = time.time()
        index = 0
        # l2 norm
        if args.l2_norm:
            l1_norm, l2_norm = torch.tensor(0., requires_grad=True).cuda(), torch.tensor(0., requires_grad=True).cuda()
        if epoch >= args.decay_epoch and epoch % args.decay_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] = p['lr'] / args.decay_lr
        for images, labels in iter(train_loader):
            images = images.cuda() 
            labels = labels.cuda() 
            optimizer.zero_grad()
            outputs = model(images)
            train_right_count = float(torch.sum(torch.eq(labels, outputs.argmax(1))))
            acc = float(train_right_count) / args.batchsize
            #loss = F.cross_entropy(outputs, labels)
            if args.use_margin:
                outputs = margin(outputs, labels, args.margin_m, args.margin_s)
            if args.dropout:
                outputs = F.dropout(outputs, p=args.dropout_rate)
            loss = F.cross_entropy(outputs, labels)
            
            if args.l2_norm:
                # l2 norm
                for p in model.fc.parameters():
                    l1_norm = l1_norm + p.norm(1) #torch.norm(p, 1).cuda()
                    l2_norm = l2_norm + p.norm(2) # torch.norm(p, 2).cuda()
                loss = loss + (l1_norm.cuda() + l2_norm.cuda())
            
            loss.backward()
            optimizer.step()
            index += 1
            if index % 10 == 0:
                print('loss:', loss.item(), 'train acc:', acc)

        test_right_count = 0.0
        for images, labels in iter(test_loader):
            images = images.cuda() 
            labels = labels.cuda() 
            outputs = model(images)
            test_right_count += float(torch.sum(torch.eq(labels, outputs.argmax(1))))

        test_accuracy = float(test_right_count) / float(len(test_dataset))
        if test_accuracy > best_accuracy:
            print('save to model')
            final_save_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH) 
            best_accuracy = test_accuracy
        t4 = time.time() - t3
        all_time += t4
        print('%d: %f, time:%f' % (epoch, test_accuracy, t4))
    print('epoch save to model', final_save_epoch, 'best acc:', test_accuracy)
    print('all train time:', all_time)

def main():
    args = parse_args()
    
    # perseus 0: init
    hvd.init()

    # perseus 1: bind gpu
    torch.cuda.set_device(hvd.local_rank())

    train(args)

if __name__ == '__main__':
    main()
