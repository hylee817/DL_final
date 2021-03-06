from torch import nn
import torch.nn.functional as F

k = 3 # kernel size
p = 3 # pool size
class CustomCNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(CustomCNN, self).__init__()
        self.input_size = input_size
        self.output_size = num_classes

        # padding = (n - ((n-k)+1)) / 2 = (k-1)/2 = 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=k, stride=1, padding=2), #in channel, out channel,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            # nn.Linear(6400, 1024),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        # out5 = self.layer5(out4)
        out = self.fc(out4)

        return out