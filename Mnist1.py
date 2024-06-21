import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import random
import pandas as pd
from arg_nene import args_parser #???
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns 
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0001
epochs = 1
num_users = 10
n_list = [40] * num_users  # Số lượng mẫu mỗi người dùng
k_list = [40] * num_users  # Số lượng mẫu mỗi lớp cho mỗi người dùng
classes_list = [np.random.choice(range(10), size=10, replace=False) for _ in range(num_users)]  # Danh sách các lớp mỗi người dùng
NUM_CLASSES = 10
def get_mnist():
    # Define data transformations
    trans_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    trans_mnist_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    # Load MNIST dataset
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=trans_mnist_train)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=trans_mnist_test)
    # Tách dữ liệu train theo tỉ lệ 9:1
    train_size = int(0.9 * len(train_dataset))
    prototype_size = len(train_dataset) - train_size
    train_dataset, prototype_dataset = random_split(train_dataset, [train_size, prototype_size])
    # Chia 90% dữ liệu cho các client
    #dict_users = mnist_noniid_lt(train_dataset, num_users, n_list, k_list, classes_list)
    #user_data_loaders = get_data_loaders(train_size, dict_users, prototype_dataset, dict_users)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)
    

    return train_loader, test_loader, prototype_loader

def get_labels(subset):
    return [subset.dataset[i][1] for i in subset.indices]
"""
def get_data_loaders(train_dataset, test_dataset, prototype_dataset, dict_users):
    # Create train loader for each client
    user_data_loaders = []
    for user in range(len(dict_users)):
        user_idx = dict_users[user]
        user_sampler = SubsetRandomSampler(user_idx)
        user_data_loader = DataLoader(train_dataset, batch_size=64, sampler=user_sampler)
        user_data_loaders.append(user_data_loader)

    # Create prototype loader for server
    prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)

    return user_data_loaders, prototype_loader"""
def mnist_noniid_lt(train_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    labels = np.array(train_dataset.dataset.targets)[train_dataset.indices]
    num_labels = len(labels)
    idxs = np.arange(num_labels)  # Chỉ lấy số phần tử tương ứng từ idxs
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    for i in range(num_users):
        k = 40
        classes = classes_list[i]
        user_data = np.array([])
        for each_class in classes:
            begin = i * 40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        user_data = user_data.astype(int)
        dict_users[i] = user_data
    return dict_users

def get_data_loaders(train_dataset,test_dataset, dict_users):
    user_data_loaders = []
    for user_id, indices in dict_users.items():
        user_sampler = SubsetRandomSampler(indices)
        user_data_loader = DataLoader(train_dataset, batch_size=64, sampler=user_sampler)
        user_data_loaders.append(user_data_loader)

    #prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)

    return user_data_loaders #, prototype_loader

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0)
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
        logits = F.log_softmax(protos, dim=1)
        #x = F.log_softmax(self.fc3(x), dim=1)
        return protos, logits

def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train_mnist_noniid(epochs, user_data_loaders, test_loader, learning_rate=0.0001):
    model = Lenet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    prototypes = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for user_loader in tqdm(user_data_loaders):
            for batch_idx, (data, target) in enumerate(user_loader):
                data, target = data.to(device), target.to(device)
                output, protos = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for j in range(target.size(0)):
                        label = target[j].item()
                        if label not in prototypes:
                            prototypes[label] = (protos[j], 1)  # Initialize prototype
                        else:
                            prototype, count = prototypes[label]
                            prototypes[label] = (prototype + protos[j], count + 1)  # Accumulate prototype

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
    }, "saved_model/LSTMModel.pt")
    # Normalize prototypes liệu nen lưu protos thành client_trainres_protos
    for label in prototypes:
        protos, count = prototypes[label]
        prototypes[label] = protos / count
    protos = {label: prototypes[label].tolist() for label in prototypes}
    return model.state_dict(), protos 

def calculate_server_prototypes(model, prototype_loader):
    server_prototypes = {}
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in enumerate(prototype_loader):
            data, target = data.to(device), target.to(device)
            output, protos = model(data)
            for j in range(protos.size(0)):
                label = j 
                if label not in server_prototypes:
                    server_prototypes[label] = (protos[j], 1)
                else:
                    prototype, count = server_prototypes[label]
                    server_prototypes[label] = (prototype + protos[j], count + 1)
        for label in server_prototypes:
            protos, count = server_prototypes[label]
            server_prototypes[label] = protos / count
        server_prototypes = {label: server_prototypes[label].tolist() for label in server_prototypes}
    return server_prototypes
            
"""
def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes):
    dist_state_dict = OrderedDict()
    for client_id, client_dict in client_trainres_protos.items():
        dist_state_dict[client_id] = {}
        for label, client_proto in client_dict.items():
            server_proto = server_prototypes[label]
            distance = np.linalg.norm(np.array(server_proto) - np.array(client_proto))
            dist_state_dict[client_id][label] = distance
    torch.save(dist_state_dict, f'distances_round_{n_round}.pt')
    torch.save(dist_state_dict, "saved_model/distance.pt")
    return dist_state_dict """

def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes ):
    dist_state_dict = OrderedDict()
    for label in range(10):
        server_proto = server_prototypes[label]
        for client_id, client_dict in client_trainres_protos.items():
            for label, protos in client_dict.items():
                client_proto = client_dict[label]
                print(client_proto)
                distance = np.linalg.norm(server_proto - client_proto)
                dist_state_dict[label][client_id]=distance
                #if label not in dist_state_dict:
                    #dist_state_dict[label] = {}
                #dist_state_dict[label][client_id] = distance
            #else:
                #print(f"Label {label} not found in client {client_id}'s data")
    torch.save(dist_state_dict, f'distances_round_{n_round}.pt')
    torch.save(dist_state_dict, "saved_model/distance.pt")
    return dist_state_dict  

"""def calculate_penalty(dist_state_dict):
    penalty_lambda = {}
    for label in dist_state_dict:
        distances = list(dist_state_dict[label].values())
        penalty = sum([1 / d for d in distances]) if len(distances) > 0 else 0.0
        penalty_lambda[label] = penalty
    return penalty_lambda"""
 
def calculate_penalty(dist_state_dict):
    penalty_lambda = {}
    for label in dist_state_dict:
        distances = list(dist_state_dict[label].values())
        penalty = sum([1 / d for d in distances]) if len(distances) > 0 else 0.0
        # Thêm các khóa tương ứng với các tham số trong mô hình
        penalty_lambda[label + '.conv1.weight'] = penalty
        penalty_lambda[label + '.conv1.bias'] = penalty
        penalty_lambda[label + '.conv2.weight'] = penalty
        penalty_lambda[label + '.conv2.bias'] = penalty
        penalty_lambda[label + '.fc1.weight'] = penalty
        penalty_lambda[label + '.fc1.bias'] = penalty
        penalty_lambda[label + '.fc2.weight'] = penalty
        penalty_lambda[label + '.fc2.bias'] = penalty
    return penalty_lambda


def start_training_task_noniid():
    # args = args_parser()
    num_users = 10  # Số lượng người dùng
    train_loader, test_loader, prototype_loader = get_mnist()
    print (prototype_loader)
    dict_users = mnist_noniid_lt(train_loader.dataset, num_users, n_list, k_list, classes_list)
    user_data_loaders = get_data_loaders(train_loader.dataset, test_loader.dataset, dict_users)
    model, prototypes = train_mnist_noniid(epochs=epochs, user_data_loaders=user_data_loaders, test_loader=test_loader, learning_rate=0.0001)
    #dist_state_dict = calculate_prototype_distance(prototypes)
    #penalty_lambda = calculate_penalty(dist_state_dict)
    # calculate_prototype_distances(prototypes)
    # print("Finish training")
    return model, prototypes
#start_training_task_noniid()