import torch

class Net(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.dropout = torch.nn.Dropout2d(0.25)
        self.fc1 = torch.nn.Linear(16 * 53 * 53, 120)
        self.fc2 = torch.nn.Linear(120, 60)
        self.fc3 = torch.nn.Linear(60, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 16 * 53 * 53)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x