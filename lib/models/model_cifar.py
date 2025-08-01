
import torch
import torch.nn as nn
import torch.nn.functional as F



# __all__ = ['MLP_top', 'mlp_top','MLP_bottom', 'mlp_bottom']



class MLP_cifar_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        self.layer_hidden = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer_hidden(x)
        return x


def mlp_cifar_top(pretrained=False, **kwargs):
    model = MLP_cifar_top(**kwargs)
    return model


class MLP_cifar_bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_input = nn.Linear(768, 50)
        # self.layer_input = nn.Linear(384, 50)
        self.dropout = nn.Dropout()  
        self.relu = nn.ReLU()  

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


def mlp_cifar_bottom(pretrained=False, **kwargs):
    model = MLP_cifar_bottom(**kwargs)
    return model


class CNN_cifar_bottom(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(int(1024), 50) 
        # self.input_dim = self._get_fc1_input_dim((3, 32, input_shape))
        self.fc1 = nn.Linear(int(2048), 50) 
        # self.fc1 = nn.Linear(self.input_dim, 50) 
        self.dropout = nn.Dropout(p=0.5)

    def _get_fc1_input_dim(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 1)
        x = torch.flatten(x,1)
        return x.shape[1]

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 1))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        return x


def cnn_cifar_bottom(pretrained=False, input_dim=32, **kwargs):
    model = CNN_cifar_bottom(input_dim, **kwargs)
    return model

class CNN_cifar_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(50*num_clients, 10)

    def forward(self, x):
        x = self.fc2(x)
        return x


class CNN_cifar100_top(nn.Module):
    def __init__(self, num_classes=100, num_clients=4):
        super().__init__()
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        # self.fc2 = nn.Linear(50*num_clients, 10)

    def forward(self, x):
        x = self.fc2(x)
        return x


def cnn_cifar100_top(pretrained=False, **kwargs):
    model = CNN_cifar100_top(**kwargs)
    return model

class LENET_cifar_bottom(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.LazyLinear(50)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        return x


def lenet_cifar_bottom(pretrained=False, **kwargs):
    model = LENET_cifar_bottom(**kwargs)
    return model

class LENET_cifar_top(nn.Module):
    def __init__(self, num_classes=10, num_clients=4):
        super().__init__()
        # self.fc2 = nn.Linear(50*num_clients, 192)
        self.fc2 = nn.Linear(50, 192)
        self.fc3 = nn.Linear(192, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet_cifar_top(pretrained=False, **kwargs):
    model = LENET_cifar_top(**kwargs)
    return model
