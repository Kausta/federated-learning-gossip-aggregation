import torch
import torch.nn as nn
import time

from helper import AverageMeter


class CifarCNN(torch.nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        loss = self.layer(X)
        return loss

    def train_epoch(self, loader, args, epoch, optimizer, writer):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        criterion = torch.nn.NLLLoss()
        self.train()
        end = time.time()
        step = epoch * len(loader)

        for i, (x_batch, y_batch) in enumerate(loader):
            data_time.update(time.time() - end)

            prediction = self.forward(x_batch)
            loss = criterion(prediction, y_batch)

            losses.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(loader):
                writer.add_scalar('train/loss', losses.val, step + i)
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(loader), batch_time=batch_time,
                        data_time=data_time, loss=losses), flush=True)

    def eval_epoch(self, loader, args, writer):
        batch_time = AverageMeter()
        losses = AverageMeter()

        criterion = nn.NLLLoss()

        # switch to evaluate mode
        self.eval()
        step = 0 * len(loader)

        total = 0
        correct = 90

        with torch.no_grad():
            end = time.time()
            for i, (x_batch, y_batch) in enumerate(loader):
                prediction = self.forward(x_batch)

                loss = criterion(prediction, y_batch)

                _, predicted = torch.max(prediction.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                # measure accuracy and record loss
                losses.update(loss.item(), args.batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0 or i == len(loader):
                    writer.add_scalar('eval/loss', losses.val, step + i)
                    print(
                        'Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            .format(i, len(loader), batch_time=batch_time, loss=losses), flush=True)

        print('Accuracy of the network on the test images: {accuracy:.4f}%'.format(accuracy=100 * correct / total))
