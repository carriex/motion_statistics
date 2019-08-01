import torch.optim as optim
import torch.nn as nn
import torch
import model
import numpy as np
import os
import constant
from dataset import UCF101DataSet
from utils import get_default_device
from tensorboardX import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def train():

    # initialize the model
    model_path = os.path.join(constant.MODEL_DIR, constant.PRETRAINED_MODEL)
    c3d = model.C3D(constant.NUM_CLASSES)

    device = get_default_device()

    if device == torch.device('cpu'):
        pretrained_param = torch.load(model_path, map_location='cpu')
    else:
        pretrained_param = torch.load(model_path)

    to_load = {}

    for key in pretrained_param.keys():
        if 'conv' in key:
            to_load[key] = pretrained_param[key]
        else:
            to_load[key] = c3d.state_dict()[key]

    c3d.load_state_dict(to_load)

    train_params = [{'params': c3d.get_1x_lr_param(), 'lr': constant.BASE_LR},
                    {'params': c3d.get_2x_lr_param(), 'lr': constant.BASE_LR * 2}]

    # import input data
    trainset = UCF101DataSet(framelist_file=constant.TRAIN_LIST, clip_len=constant.CLIP_LENGTH,
                             crop_size=constant.CROP_SIZE, split="training")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=constant.TRAIN_BATCH_SIZE,
                                              shuffle=True, num_workers=10)

    c3d.to(device, non_blocking=True, dtype=torch.float)
    c3d.train()

    # define loss function (Cross Entropy loss)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # define optimizer
    optimizer = optim.SGD(train_params, lr=constant.BASE_LR,
                          momentum=constant.MOMENTUM, weight_decay=constant.WEIGHT_DECAY)

    # define lr schedule

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=constant.LR_DECAY_STEP_SIZE,
                                          gamma=constant.LR_DECAY_GAMMA)
    writer = SummaryWriter()

    for epoch in range(constant.NUM_EPOCHES):

        running_loss = 0.0
        running_accuracy = 0.0
        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            step = epoch * len(trainloader) + i
            inputs, labels = data['clip'].to(device, dtype=torch.float), data['label'].to(
                device=device, dtype=torch.int64)
            optimizer.zero_grad()

            outputs = c3d(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('Step %d, loss: %.3f' % (i, loss.item()))
            writer.add_scalar('Train/Loss', loss.item(), step)

            outputs = nn.Softmax(dim=1)(outputs)
            _, predict_label = outputs.max(1)
            correct = (predict_label == labels).sum().item()
            accuracy = float(correct) / float(constant.TRAIN_BATCH_SIZE)
            running_accuracy += accuracy
            writer.add_scalar('Train/Accuracy', accuracy, step)

            print("iteration %d, accuracy = %.3f" % (i, accuracy))

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                print('[%d, %5d] accuracy: %.3f' %
                      (epoch + 1, i + 1, running_accuracy / 100))
                running_loss = 0.0
                running_accuracy = 0.0
            if step % 10000 == 9999:
                torch.save(c3d.state_dict(), os.path.join(
                    model_dir, '%s-%d' % (constant.TRAIN_MODEL_NAME, step+1)))

    print('Finished Training')
    writer.close()


def main():
    train()


if __name__ == "__main__":
    main()
