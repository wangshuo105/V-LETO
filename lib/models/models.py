import torch
import torch.nn as nn
import torch.nn.functional as F



# __all__ = ['MLP_top', 'mlp_top','MLP_bottom', 'mlp_bottom']



class MLP_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        # self.layer_hidden = nn.Linear(50*num_clients, 10)
        self.layer_hidden = nn.Linear(50, 10)

    def forward(self, x):
        x = self.layer_hidden(x)
        return x


def mlp_top(pretrained=False, **kwargs):
    model = MLP_top(**kwargs)
    return model


class MLP_bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_input = nn.Linear(196, 50)
        self.dropout = nn.Dropout()  
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


def mlp_bottom(pretrained=False, **kwargs):
    model = MLP_bottom(**kwargs)
    return model

class MLP5(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP5, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

def MLP5_bottom(input_dim, output_dim, hidden_dim=128):
    model = MLP5(input_dim, output_dim, hidden_dim)
    return model



class MLP5_top(nn.Module):

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc1_top = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2_top = nn.Linear(hidden_dim//2, num_classes)

    def forward(self, z_list):

        x = self.fc1_top(z_list)
        x = F.relu(x)
        x = self.fc2_top(x)
        x = F.relu(x)
        x = x.squeeze(1)
        return x

def mlp5_top(hidden_dim, num_classes):
    model = MLP5_top(hidden_dim, num_classes)
    return model


class CNN_bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(1344), 50) 
        # self.fc1 = nn.Linear(int(896), 50) 


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        return x


def cnn_bottom(pretrained=False, **kwargs):
    model = CNN_bottom(**kwargs)
    return model

class CNN_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        self.fc2 = nn.Linear(50, 192)
        self.fc3 = nn.Linear(192, 160)
        self.fc4 = nn.Linear(160, 10) 
        # self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def cnn_top(pretrained=False, **kwargs):
    model = CNN_top(**kwargs)
    return model

class LENET_bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.fc1 = None
        self.fc1 = nn.Linear(4608, 50)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        return x
    

def lenet_bottom(pretrained=False, **kwargs):
    model = LENET_bottom(**kwargs)
    return model

class LENET_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        self.fc2 = nn.Linear(50, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet_top(pretrained=False, **kwargs):
    model = LENET_top(**kwargs)
    return model
