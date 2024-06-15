import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import time
import psutil
import json
import random
import pandas as pd
from arg_nene import args_parser #???
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
epochs = 1
num_users = 10
n_list = [40] * num_users  # Số lượng mẫu mỗi người dùng
k_list = [40] * num_users  # Số lượng mẫu mỗi lớp cho mỗi người dùng
classes_list = [np.random.choice(range(10), size=10, replace=False) for _ in range(num_users)]  # Danh sách các lớp mỗi người dùng

def get_mnist():
    # Define data transformations
    trans_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trans_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Load MNIST dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=trans_mnist_train)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=trans_mnist_test)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    return train_loader, test_loader

# noniid
def mnist_noniid_lt(train_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40
        classes = classes_list[i]
        # print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        # Ép kiểu user_data thành số nguyên
        user_data = user_data.astype(int)
        dict_users[i] = user_data
    return dict_users

def get_data_loaders(dataset, dict_users):
    user_data_loaders = []
    for user in range(len(dict_users)):
        user_idx = dict_users[user]
        user_sampler = torch.utils.data.SubsetRandomSampler(user_idx)
        user_data_loader = DataLoader(dataset, batch_size=64, sampler=user_sampler)
        user_data_loaders.append(user_data_loader)
    return user_data_loaders

"""class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=32*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        protos = self.fc3(out)
        out = F.log_softmax(self.fc3(out), dim=1)
        return out, protos"""

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        protos = self.fc3(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return protos, x

def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train_mnist_noniid(epochs, user_data_loaders, test_loader, learning_rate=0.0001):
    model = Lenet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    prototypes ={}   # {i: (torch.zeros(192).to(device), 0) for i in range(10)}

    for epoch in range(1, epochs + 1):  
        model.train()
        for user_loader in tqdm(user_data_loaders):
            for batch_idx, (data, target) in enumerate(user_loader):
                data, target = data.to(device), target.to(device)
                batch_idx = int(batch_idx)
                output, protos = model(data)
                loss = criterion(output, target)  
                loss.backward()
                optimizer.step()
                # Cập nhật prototypes sau mỗi batch
                with torch.no_grad():
                    for j in range(target.size(0)):
                        label = target[j].item()
                        if label not in prototypes:
                            prototypes[label] = (protos[j], 0)  
                        prototype, count = prototypes[label]
                        prototypes[label] = (prototype, count + 1)       
        # Kiểm tra trên tập kiểm tra sau mỗi epoch
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}%')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'accuracy': accuracy,
        'prototypes': prototypes
    }, 'mymodel.pt')
    for label in prototypes:
        protos, count = prototypes[label]
        prototypes[label] = protos / count
        # print(f'Label: {label}, Prototype: {prototypes[label]}, Count: {count}')
    
    protos = {label: prototypes[label] for label in prototypes}
    protos = {label: tensor_to_list(prototypes[label]) for label in prototypes}
    # prototypes_json = {label: tensor_to_list(prototypes[label]) for label in prototypes}
    # print(prototypes)
    return model.state_dict(), protos

def calculate_prototype_distances(prototypes):
    labels = sorted(prototypes.keys())
    protos = [prototypes[label] for label in labels]
    proto_tensor = torch.stack(protos)    # Chuyển list các prototype thành tensor
    #dist_matrix = torch.nn.functional.pairwise_distance(proto_tensor, proto_tensor)
    dist_matrix = torch.cdist(proto_tensor.unsqueeze(0), proto_tensor.unsqueeze(0), p=2)
    print("Distance Matrix:")
    print(dist_matrix)
    return dist_matrix, labels 

def start_training_task_noniid():
    # args = args_parser()
    num_users = 10  # Số lượng người dùng
    train_loader, test_loader = get_mnist()
    dict_users = mnist_noniid_lt(test_loader.dataset, num_users, n_list, k_list, classes_list)
    user_data_loaders = get_data_loaders(train_loader.dataset, dict_users)
    model, prototypes = train_mnist_noniid(epochs=epochs, user_data_loaders=user_data_loaders, test_loader=test_loader, learning_rate=0.0001)
    # calculate_prototype_distances(prototypes)
    # print("Finish training")
    return model, prototypes

# start_training_task_noniid()




