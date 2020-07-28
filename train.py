import os
import torch.optim as optim
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
import model
import constant
import numpy as np
from dataset import UCF101DataSet
from utils import get_default_device

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def train():
    # initialize the model
    c3d = model.C3D(constant.NUM_MOTION_LABEL, pretrain=True)

    train_param = [{'params': c3d.get_fc_1x_lr_param(), 'weight_decay': constant.WEIGHT_DECAY},
                   {'params': c3d.get_fc_2x_lr_param(), 'lr': constant.BASE_LR * 2},
                   {'params': c3d.get_conv_1x_lr_param()},
                   {'params': c3d.get_conv_2x_lr_param(), 'lr': constant.BASE_LR * 2}]

    device = get_default_device()

    # import input data
    trainset = UCF101DataSet(framelist_file=constant.FRAMELIST_FILE,
                             v_flow_list_file=constant.V_FLOW_LIST_FILE,
                             u_flow_list_file=constant.U_FLOW_LIST_FILE,
                             clip_len=constant.CLIP_LENGTH,
                             crop_size=constant.CROP_SIZE,
                             split="training")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=constant.TRAIN_BATCH_SIZE,
                                              shuffle=True, num_workers=10)
    c3d.train()
    c3d.to(device, non_blocking=True, dtype=torch.float)

    # define loss function (MSE loss)
    criterion = nn.MSELoss()
    criterion.to(device)

    # define optimizer
    optimizer = optim.SGD(train_param, lr=constant.BASE_LR,
                          momentum=constant.MOMENTUM)

    # lr is divided by 10 after every 4 epoches
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=constant.LR_DECAY_STEP_SIZE,
                                          gamma=constant.LR_DECAY_GAMMA)

    writer = SummaryWriter()

    print(constant.PRETRAIN_MODEL_NAME)

    for epoch in range(constant.NUM_EPOCHES):

        scheduler.step()

        for i, data in enumerate(trainloader, 0):
            step = epoch * len(trainloader) + i
            inputs, labels = data['clip'].to(
                device, dtype=torch.float), data['label'].to(device)
            optimizer.zero_grad()

            outputs = c3d(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))
            if step % 10000 == 9999:
                torch.save(c3d.state_dict(),
                           os.path.join(constant.MODEL_DIR, '%s-%d' % (constant.PRETRAIN_MODEL_NAME, step + 1)))

    print('Finished Training')
    writer.close()


def main():
    train()


if __name__ == "__main__":
    main()
