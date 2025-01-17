import torch.nn as nn
import torch.nn.functional as F


# input: N x 3 x  16 x 112 x 112
# N X C X D X H X W
# torch.nn.conv3d(in_channels, out_channels, kernel_size, stride=1,padding=0)


class C3D(nn.Module):
    def __init__(self, num_classes, pretrain=False):
        super(C3D, self).__init__()

        self.pretrain = pretrain

        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)

        self.pool1 = nn.MaxPool3d((1, 2, 2), ceil_mode=True)

        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)

        self.pool2 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)

        self.conv3 = nn.Conv3d(128, 256, 3, padding=1)

        self.pool3 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)

        self.conv4 = nn.Conv3d(256, 256, 3, padding=1)

        self.pool4 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)

        self.conv5 = nn.Conv3d(256, 256, 3, padding=1)

        self.pool5 = nn.MaxPool3d((2, 2, 2), ceil_mode=True)

        self.fc1 = nn.Linear(256 * 1 * 4 * 4, 2048)  # 2048/4096

        self.fc2 = nn.Linear(2048, 2048)

        # num of classes are output dimension for pretrain
        self.out = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.init_weights()

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 1 * 4 * 4)
        x = F.relu(self.fc1(x))
        if self.pretrain == False:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.pretrain == False:
            x = self.dropout(x)
        x = self.out(x)

        return x

    def init_weights(self):

        for name, m in self.named_modules():
            if type(m) == nn.Conv3d:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)
            if type(m) == nn.Linear:
                if name == 'out':
                    nn.init.constant_(m.bias, 0.0)
                    if self.pretrain == True:
                        nn.init.normal_(m.weight, std=0.01)
                    else:
                        nn.init.normal_(m.weight, std=0.005)
                else:
                    nn.init.constant_(m.bias, 1.0)
                    nn.init.normal_(m.weight, std=0.005)

    def get_conv_1x_lr_param(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'conv' in name and param.requires_grad:
                yield param

    def get_conv_2x_lr_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name and 'conv' in name and param.requires_grad:
                yield param

    def get_fc_1x_lr_param(self):
        for name, param in self.named_parameters():
            if 'conv' not in name and 'weight' in name and param.requires_grad:
                yield param

    def get_fc_2x_lr_param(self):
        for name, param in self.named_parameters():
            if 'conv' not in name and 'bias' in name and param.requires_grad:
                yield param
