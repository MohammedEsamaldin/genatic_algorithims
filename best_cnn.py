import os
import torch
import librosa
import numpy as np
import torchvision
import scipy.io
from scipy.signal.windows import hann
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 30

# You only need to define your best model here. The rest of the code will work as is
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.output = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

# data_path = "/Users/mohammed/Desktop/AI/ECS759P_CW2/FashionMNIST/raw"

train_set = torchvision.datasets.FashionMNIST(root = ".", train=True,
                                              download=True,
                                              transform=transforms.ToTensor())
validation_set = torchvision.datasets.FashionMNIST(root = ".", train=False,
                                             download=True,
                                             transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)
# Fix the seed to be able to get the same randomness across runs and
# hence reproducible outcomes
torch.manual_seed(0)


best_model = Net()  # Instantiate the model
state_dict = torch.load("/Users/mohammed/Desktop/AI/ECS759P_CW2/best_cnn.pth",weights_only=True)  # Load the state dictionary
# filtered_state_dict = {k: v for k, v in state_dict.items() if k in best_model.state_dict()}
best_model.load_state_dict(state_dict)


# Evaluate the model on the validation set
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in validation_loader:
        # print(target)
        output = best_model(data)
        pred = output.argmax(dim=1, keepdim=True)
        target = target.view(-1, 1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

validation_accuracy = correct / total
print(f'Evaluation on validation set complete. Accuracy: {validation_accuracy:.4f} ({correct}/{total} correct)')