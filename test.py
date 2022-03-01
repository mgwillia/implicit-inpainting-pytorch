import argparse
import numpy as np
import torch
from models import resnet12

from datasets import MiniImageNet
from torchvision import transforms


def eval(model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4,
            batch_size=4096, pin_memory=True, drop_last=False, shuffle=False)

    num_correct = 0
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            images = batch['image'].cuda(non_blocking=True)
            targets = batch['target'].cuda(non_blocking=True)
            outputs = model(images)
            num_correct += torch.eq(outputs.argmax(1), targets).float().sum()
    print(f'Val accuracy: {num_correct / len(dataset)}')
    model.train()


def main():
    ### SET UP PARSER ###
    parser = argparse.ArgumentParser(description='Get Features')
    parser.add_argument('--learning-rate', default=3.0, type=float, help='learning rate')
    args = parser.parse_args()

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(84),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #train_dataset_full = MiniImageNet('/scratch0/mgwillia/mini-ImageNet/', split='train', transform=val_transform)
    #train_idxs = [each for each in range(len(train_dataset_full)) if each % 5 != 4]
    #val_idxs = [each for each in range(len(train_dataset_full)) if each % 5 == 4]
    #train_dataset_train = torch.utils.data.Subset(train_dataset_full, train_idxs)
    #train_dataset_val = torch.utils.data.Subset(train_dataset_full, val_idxs)
    train_dataset_train = MiniImageNet('/scratch0/mgwillia/mini-ImageNet/', split='train', transform=train_transform)
    train_dataset_val = MiniImageNet('/scratch0/mgwillia/mini-ImageNet/', split='train', transform=val_transform)

    num_epochs = 350
    lr_init = args.learning_rate
    model = resnet12(num_classes=64).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,300],gamma=0.1)

    train(model, train_dataset_train, train_dataset_val, optimizer, criterion, scheduler, num_epochs)
    eval(model, train_dataset_val)
    torch.save(model.state_dict(), f'r12_mininet_{str(lr_init).replace(".", "_")}.pth.tar')

if __name__ == '__main__':
    main()
