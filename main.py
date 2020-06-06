import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchsummary import summary
import numpy as np
from CifarCNN import CifarCNN
from GossipAggregator import GossipAggregator
from communication.rpc_server import server_from_peers_file


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def main():
    print("Cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("Using device", dev)
    device = torch.device(dev)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--id', default="test", type=str)
    parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128, help="Training data batch size")
    parser.add_argument('--data_dir', type=str, default="./data", help="Data directory")
    parser.add_argument("--gossip", type=bool, default=False, help="Gossip mode")
    parser.add_argument("--indices", type=str, default=None, help="Indices file")
    parser.add_argument("--peers", type=str, default="peers.txt", help="Peers file")
    args = parser.parse_args()

    api = None
    if args.gossip:
        peers = args.peers
        api = server_from_peers_file(peers)

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

    if args.indices is not None:
        indices = torch.load(args.indices)
    else:
        indices = list(range(len(train_set)))
    dataset = torch.utils.data.Subset(train_set, indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_train, shuffle=True, num_workers=1, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)

    run_dir = os.path.join("./", "runs", args.id)
    writer = SummaryWriter(log_dir=run_dir)

    model = CifarCNN(device)
    summary(model, (3, 32, 32))
    gossip = None
    if args.gossip:
        gossip = GossipAggregator(data_points=len(indices), server_api=api)

    optimizer = optim.AdamW(model.parameters(), 0.001, betas=(0.9, 0.999), weight_decay=1e-2)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for epoch in range(200):
        print("Training epoch:", epoch)
        model.train_epoch(train_loader, args, epoch, optimizer, writer)
        print("Evaluating epoch:", epoch)
        model.eval_epoch(test_loader, args, writer)
        lr_scheduler.step()
        if args.gossip:
            flattened = model.flatten()
            print("Pushing updates:", epoch)
            gossip.push_model(flattened)
            print("Receiving updates:", epoch)
            flattened = gossip.receive_updates(flattened)
            model.unflatten(flattened)
            print("Evaluating post receive:", epoch)
            model.eval_epoch(test_loader, args, writer)


if __name__ == '__main__':
    main()
