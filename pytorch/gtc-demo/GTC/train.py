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
from LossFunction import FocalLoss, margin
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
    parser.add_argument('--start-freeze-epoch', type=int, default=0, help='start unfreeze layer after epoch')
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
    parser.add_argument('--color-jit', type=int, default=1, help='1: use colorjit to enhance data')
    parser.add_argument('--random-gray', type=int, default=1, help='1: use random-gray to enhance data')
    parser.add_argument('--use-margin', type=int, default=1, help='1: use margin loss')
    parser.add_argument('--margin-m', type=float, default=0.35, help='cosine margin m')
    parser.add_argument('--margin-s', type=float, default=30., help='feature scale s')
    parser.add_argument('--use-focal-loss', type=int, default=1, help='1: use focal loss')
    parser.add_argument('--use-outside-data', type=int, default=0, help='1: use outside data')

    args = parser.parse_args()
    return args


def train(args):
    t0 = time.time()

    dataset = datasets.ImageFolder(
        'dataset',
        transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomGrayscale(p=0.5) if args.random_gray else transforms.RandomGrayscale(p=0.0),
            #transforms.ColorJitter(0.1, 0.1, 0.1, 0.1) if args.color_jit else '',
            #transforms.ColorJitter(brightness=1) if args.color_jit else '', 
            #transforms.ColorJitter(contrast=1) if args.color_jit else '',
            #transforms.ColorJitter(saturation=0.5) if args.color_jit else '',
            #transforms.ColorJitter(hue=0.5) if args.color_jit else '',
            transforms.Resize((224, 224)),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # data loader
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - args.test_size, args.test_size])
    
    if args.use_outside_data:
        test_dataset = datasets.ImageFolder(
            'test-mini-dataset',
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        sampler=train_sampler,
        num_workers=args.prefetch_nums,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.prefetch_nums,
        drop_last=True
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
    
    def frozen():
        # frozen part layer of model
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
     
    if not args.start_freeze_epoch:
        frozen()

    # perseus 2: model to cuda
    model = model.cuda() 

    NUM_EPOCHS = args.end_epoch
    BEST_MODEL_PATH = args.save_model 
    best_accuracy = 0.0

    # perseus 3: lr * hvd.size()
    lr = args.lr * hvd.size()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.01)
    
    # perseus 4: wrapper optimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # perseus 5: broadcast model parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # perseus 6: broadcast opt state
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    t2 = time.time()
    all_time = 0

    # focal loss
    if args.use_focal_loss:
        print('use focal loss')
        #criterion = FocalLoss(alpha=[1,1,1], gamma=2, num_classes=args.num_classes)
        criterion = FocalLoss(gamma=2)
    else:
        criterion = F.cross_entropy 

    model.train()
    final_save_epoch = 0
    for epoch in range(NUM_EPOCHS):
        t3 = time.time()
        index = 0
        # l2 norm
        if args.l2_norm:
            assert "replace l2_norm with optimizer's weight_decay"
            l1_norm, l2_norm = torch.tensor(0., requires_grad=True).cuda(), torch.tensor(0., requires_grad=True).cuda()
        if epoch >= args.decay_epoch and epoch % args.decay_epoch == 0:
            for p in optimizer.param_groups:
                p['lr'] = p['lr'] / args.decay_lr

        if epoch > 0 and epoch == args.start_freeze_epoch:
            frozen()

        train_right_count_all = 0.0
        average_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.cuda() 
            labels = labels.cuda() 
            optimizer.zero_grad()
            outputs = model(images)
            train_right_count = float(torch.sum(torch.eq(labels, outputs.argmax(1))))
            train_right_count_all += train_right_count
            acc = float(train_right_count) / len(labels) #args.batchsize
            #loss = F.cross_entropy(outputs, labels)
            if args.use_margin:
                outputs = margin(outputs, labels, args.margin_m, args.margin_s)
            if args.dropout:
                outputs = F.dropout(outputs, p=args.dropout_rate)
            loss = criterion(outputs, labels) # F.cross_entropy(outputs, labels)
            
            if args.l2_norm:
                # l2 norm 
                for p in model.fc.parameters():
                    l1_norm = l1_norm + p.norm(1) #torch.norm(p, 1).cuda()
                    l2_norm = l2_norm + p.norm(2) # torch.norm(p, 2).cuda()
                loss = loss + (l1_norm.cuda() + l2_norm.cuda())
            
            loss.backward()
            optimizer.step()
            index += 1
            average_loss += loss.item()
            if index % 100 == 0:
                print('epoch:', epoch, 'step:', index, 'loss:', loss.item(), 'train acc:', acc)
        average_loss = float(average_loss) / float(index)
        train_accuracy = float(train_right_count_all) / float(index * args.batchsize)

        # test
        test_right_count = 0.0
        for images, labels in iter(test_loader):
            images = images.cuda() 
            labels = labels.cuda() 
            outputs = model(images)
            test_right_count += float(torch.sum(torch.eq(labels, outputs.argmax(1))))

        test_accuracy = float(test_right_count) / float(args.batchsize * len(test_loader))
        if test_accuracy >= best_accuracy:
            print('save to model, best acc:{}'.format(test_accuracy))
            final_save_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH) 
            best_accuracy = test_accuracy
        t4 = time.time() - t3
        all_time += t4
        print('epoch:%2d, average_loss:%f, train_acc:%f, test_acc:%f, time:%f' % (epoch, average_loss, train_accuracy, test_accuracy, t4))
    print('at {} epoch save to model, best acc:{}'.format(final_save_epoch, best_accuracy))
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
