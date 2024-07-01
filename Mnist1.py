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
NUM_DEVICE = 2
n_list = [40] * NUM_DEVICE    # Số lượng mẫu mỗi người dùng
k_list = [40] * NUM_DEVICE  # Số lượng mẫu mỗi lớp cho mỗi người dùng
classes_list = [np.random.choice(range(10), size=10, replace=False) for _ in range(NUM_DEVICE)]  # Danh sách các lớp mỗi người dùng
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
    dict_users = mnist_noniid_lt(train_dataset, NUM_DEVICE)
    user_data_loaders = get_data_loaders(train_dataset, dict_users)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)
    

    return train_loader, test_loader, prototype_loader, train_dataset

def get_labels(subset):
    return [subset.dataset[i][1] for i in subset.indices]

def mnist_noniid_lt(train_dataset, NUM_DEVICE, n_list=None):

    dict_users = {i: np.array([], dtype='int64') for i in range(NUM_DEVICE)}
    labels = np.array(train_dataset.dataset.targets)[train_dataset.indices]
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # Ensure each client has at least one sample from each class
    min_samples_per_class = 1
    NUM_CLASSES = len(np.unique(labels))
    for label in range(NUM_CLASSES):
        label_idxs = idxs[idxs_labels[1] == label]
        np.random.shuffle(label_idxs)
        for i in range(NUM_DEVICE):
            selected_idxs = label_idxs[i * min_samples_per_class: (i + 1) * min_samples_per_class]
            dict_users[i] = np.concatenate((dict_users[i], selected_idxs), axis=0)
    
    # Distribute remaining samples
    remaining_idxs = idxs[NUM_DEVICE * NUM_CLASSES:]
    np.random.shuffle(remaining_idxs)

    # Calculate the number of samples for each client
    samples_per_client = len(remaining_idxs) // NUM_DEVICE
    for i in range(NUM_DEVICE):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i != NUM_DEVICE - 1 else len(remaining_idxs)
        dict_users[i] = np.concatenate((dict_users[i], remaining_idxs[start_idx:end_idx]), axis=0)
    for i in range(NUM_DEVICE):
        print(f"Client {i} has {len(dict_users[i])} samples.")
    return dict_users

def get_data_loaders(train_dataset, dict_users):
    user_data_loaders = []
    for client_id, indices in dict_users.items():
        if client_id in dict_users:
            user_sampler = SubsetRandomSampler(indices)
            user_data_loader = DataLoader(train_dataset, batch_size=64, sampler=user_sampler)
            user_data_loaders.append(user_data_loader)
        else:
            print(f"Client {client_id} không thuộc dict_users.")
    return user_data_loaders

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
        output = F.log_softmax(protos, dim=1)
        #x = F.log_softmax(self.fc3(x), dim=1)
        return output, protos

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
                            prototypes[label] = (protos[j], 1)  
                        else:
                            prototype, count = prototypes[label]
                            prototypes[label] = (prototype + protos[j], count + 1) 
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
    # Normalize prototypes 
    for label in prototypes:
        protos, count = prototypes[label]
        prototypes[label] = protos / count
    protos = {label: prototypes[label].tolist() for label in prototypes}
    return model.state_dict(), protos 

def calculate_server_prototypes(model, prototype_loader):
    server_prototypes = {}
    model = Lenet()
    model.load_state_dict(torch.load("saved_model/LSTMModel.pt")['model_state_dict'])
    model.to(device)
    model.eval()  
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(prototype_loader):
            data, target = data.to(device), target.to(device)
            output, protos = model(data)
            for j in range(protos.size(0)):
                label = target[j].item()  
                if label not in server_prototypes:
                    server_prototypes[label] = (protos[j], 1)
                else:
                    prototype, count = server_prototypes[label]
                    server_prototypes[label] = (prototype + protos[j], count + 1)
    for label in server_prototypes:
        protos, count = server_prototypes[label]
        server_prototypes[label] = protos / count
    server_prototypes = {label: server_prototypes[label].tolist() for label in server_prototypes}
    torch.save(server_prototypes, 'server_prototypes.pt')
    
    return server_prototypes

"""
def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes):
    dist_state_dict = OrderedDict()
    #print("client trainres proto:", client_trainres_protos)
    #print("server proto:", server_prototypes)

    for label in range(10):  # Duyệt qua các nhãn từ 0 đến 9
        server_proto = np.array(server_prototypes[label])
        for client_id, client_dict in client_trainres_protos.items():
            for label, protos in client_dict.items(): 
                print("{0}: ({1} : {2})".format(client_id, label,protos))
                client_proto = np.array(client_dict[label])
                distance = np.linalg.norm(server_proto - client_proto)
                if label not in dist_state_dict:
                    dist_state_dict[label] = {}
                dist_state_dict[label][client_id] = distance
                
            #print(f"Client {client_id} does not have label {label}")
            #print(f"Available labels for client {client_id}: {list(client_dict.keys())}")
    torch.save(dist_state_dict, f'distances_round_{n_round}.pt')
    torch.save(dist_state_dict, "saved_model/distance.pt")
    return dist_state_dict """
"""
def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes):
    #dist_state_dict = OrderedDict()
    server_proto = []
    clients_proto = {}
    distance = OrderedDict()
    for label, proto_server in server_prototypes.items():
            #print("label_server",label)
            server_proto.append(np.array(server_prototypes[label]))
    for client_id, client_protodict in client_trainres_protos.items():
        protos = []
        for label, proto in client_protodict.items():
            #print("label client:", label)
            protos.append(np.array(proto))
            clients_proto[client_id] = np.array(protos)
            protos = []
                    
    print("\n Proto tren Server: ", server_proto)
    print("\n Proto tren Client: ", clients_proto)
    for client_id, protos in clients_proto.items():
        for label in range(len(server_proto)):
            #server_proto = np.array(server_prototypes[label])
            #clients_proto = np.array(clients_proto[label])
            distance[client_id] = np.linalg.norm(server_proto - clients_proto[client_id])

    return distance"""
def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes):
    server_proto = {}
    clients_proto = {}
    dist_state_dict = OrderedDict()
    for label, proto_server in server_prototypes.items():
        server_proto[label] = np.array(proto_server)
        #server_proto.append(np.array(server_prototypes[label]))
    for client_id, client_protodict in client_trainres_protos.items():
        protos = {}
        for label, proto in client_protodict.items():
            #protos.append(np.array(proto))
            protos[label] = np.array(proto)
        clients_proto[client_id] = protos
    print("\n Proto tren Server: ", server_proto)
    print("\n Proto tren Client: ", clients_proto)
    for client_id, protos in clients_proto.items():
        client_distances = {}
        for label in server_proto.keys():
            if label in protos:
                distance = np.linalg.norm(server_proto[label] - protos[label])
                client_distances[label] = distance
                print(f"Client: {client_id}, Label: {label}, Server Proto Shape: {server_proto[label].shape}, Client Proto Shape: {protos[label].shape}, Distance: {distance}")
            else:
                client_distances[label] = None  # Nếu client không có prototype cho nhãn này
        dist_state_dict[client_id] = client_distances 
        """
    for client_id, protos in clients_proto.items():
        client_distances = {}
        for label in range(len(server_proto)):
            if label in protos:
                client_distances[label] = np.linalg.norm(server_proto[label] - protos[label])
            else:
                client_distances[label] = None 
        dist_state_dict[client_id] = client_distances """
    return dist_state_dict

def calculate_penalty(dist_state_dict):
    penalty_lambda = {}
    for client_id, distances_dict in dist_state_dict.items():
        client_penalty = {}
        for label, distance in distances_dict.items():
            if distance is not None and distance != 0:
                client_penalty[label] = 1 / distance
            else:
                client_penalty[label] = 0  
        penalty_lambda[client_id] = client_penalty
    return penalty_lambda

"""
def start_training_task_noniid():
    # args = args_parser()
    NUM_DEVICE = 2  # Số lượng người dùng
    train_loader, test_loader, prototype_loader, train_dataset = get_mnist()
    print (prototype_loader)
    dict_users = mnist_noniid_lt(train_dataset, NUM_DEVICE)
    user_data_loaders = get_data_loaders(train_loader.dataset, test_loader.dataset, dict_users)
    model, prototypes = train_mnist_noniid(epochs=epochs, user_data_loaders=user_data_loaders, test_loader=test_loader, learning_rate=0.0001)
    #dist_state_dict = calculate_prototype_distance(prototypes)
    #penalty_lambda = calculate_penalty(dist_state_dict)
    # print("Finish training")
    return model, prototypes"""
def start_training_task_noniid():
    train_loader, test_loader, prototype_loader, train_dataset = get_mnist()
    dict_users = mnist_noniid_lt(train_dataset, NUM_DEVICE)
    user_data_loaders = get_data_loaders(train_dataset, dict_users)
    model, prototypes = train_mnist_noniid(epochs=epochs, user_data_loaders=user_data_loaders, test_loader=test_loader, learning_rate=0.0001)
    return model, prototypes
#start_training_task_noniid()