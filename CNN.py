from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(20 * 128 * 128, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 20 * 128 * 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

    # def __init__(self):
    #     super(CNNModel, self).__init__()
    #
    #     self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1) #in / out(filters) / filter_size / stride
    #     self.pool = nn.MaxPool2d(stride=(2,2), kernel_size=(2,2))
    #     self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
    #     self.dropout025 = nn.Dropout(0.25)
    #     self.fc1 = nn.Linear(in_features=)
    #     self.fc2 = nn.Linear()
    #     self.fc3 = nn.Linear()
    #     self.softmax = nn.LogSoftmax(dim=1)
    #
    #     self.dropout05 = nn.Dropout(0.5)
    #     self.dense = nn.Linear(in_features=64*128, out_features=512)
    #     self.pred = nn.Linear(in_features=512, out_features=2)
    #     self.softmax = nn.Softmax(dim=1)
    #
    #     self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
    #     self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    #     self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    #     self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
    #
    #     self.batch1 = nn.BatchNorm2d(16)
    #     self.batch2 = nn.BatchNorm2d(32)
    #     self.batch3 = nn.BatchNorm2d(64)
    #     self.batch4 = nn.BatchNorm2d(128)
    #     self.batch5 = nn.BatchNorm2d(256)
    #
    #
    # def forward(self, x):
    #     # cnn2d -> relu -> maxpool2d -> dropout
    #     x = self.pool(F.relu(self.conv1(x)))
    #     # x = self.dropout025(x)
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.dropout025(x)
    #     # x = self.pool(F.relu(self.conv3(x)))
    #     # x = self.dropout025(x)
    #     x = self.pool(F.relu(self.conv4(x)))
    #     x = self.dropout025(x)
    #     # x = self.pool(F.relu(self.conv5(x)))
    #     # x = self.dropout025(x)
    #     # x => torch.Size([5, 256, 2, 30)] => torch.Size([5, 256, 4, 32)]
    #
    #     # global pooling and MLP
    #     x = x.view(x.size(0), -1) # = flatten in keras
    #     x = self.dropout05(x)
    #     x = F.relu(self.dense(x))
    #     x = self.dropout025(x)
    #     predictions = self.pred(x) #supposed to be [batch_size, classes]
    #     # predictions = self.softmax(self.pred(x))
    #     # print(predictions)
    #
    #     return predictions
