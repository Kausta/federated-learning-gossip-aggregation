import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchsummary import summary

from CifarCNN import CifarCNN


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--id', default="test", type=str)
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256, help="Training data batch size")
    parser.add_argument('--data_dir', type=str, default="./data", help="Data directory")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    args = parser.parse_args()

    batch_size_train = args.batch_size
    batch_size_test = args.batch_size

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False, num_workers=2)

    run_dir = os.path.join("./", "runs", args.id)
    writer = SummaryWriter(log_dir=run_dir)

    model = CifarCNN()
    summary(model, (3, 32, 32))
    epochs = args.epochs
    optimizer = optim.Adam(
        model.parameters(), 0.001, betas=(0.9, 0.999))
    for epoch in range(epochs):
        model.train_epoch(train_loader, args, epoch, optimizer, writer)
        model.eval_epoch(test_loader, args, writer)


if __name__ == '__main__':
    main()
