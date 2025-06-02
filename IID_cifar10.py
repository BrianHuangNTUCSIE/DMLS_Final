import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import argparse
import csv
from tqdm import tqdm

# ----- Model Definitions for CIFAR-10 -----
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),  # (N, 3, 32, 32) -> (N, 6, 32, 32)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # (N, 6, 16, 16)
            nn.Conv2d(6, 16, kernel_size=5, stride=1),            # (N, 16, 12, 12)
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # (N, 16, 6, 6)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_model(model_type):
    if model_type == "LR":
        return LR()
    elif model_type == "DNN":
        return DNN()
    elif model_type == "CNN":
        return CNN()
    elif model_type == "DenseNet":
        model = models.densenet121(num_classes=10)
        return model
    else:
        raise ValueError("Unknown model type. Choose from: LR, DNN, CNN, DenseNet.")

# ----- Load and Split CIFAR-10 -----
def load_data(delta):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    total_data = len(train_dataset)  # Should be 50000
    group_size = total_data // 10
    indices = np.arange(total_data)
    np.random.shuffle(indices)

    # Step 1: Split into 10 original groups
    original_groups = [indices[i * group_size:(i + 1) * group_size] for i in range(10)]

    client_datasets = []
    shared_per_group = int(group_size * delta / 10)
    for client_id in range(10):
        shared_data = []
        # Step 3: Pull shared data from all 10 groups
        for group in original_groups:
            shared_data.extend(group[:shared_per_group])
        shared_data = shared_data[:6000]  # Limit in case delta makes it >6000

        # Step 4: Add private data if needed
        if len(shared_data) < 6000:
            remaining = 6000 - len(shared_data)
            own_group = original_groups[client_id]
            used_shared_indices = set(shared_data)
            private_candidates = [i for i in own_group if i not in used_shared_indices]
            shared_data.extend(private_candidates[:remaining])

        client_datasets.append(Subset(train_dataset, shared_data))

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return client_datasets, test_loader

# ----- Local Training -----
def train_local(model, dataloader, device):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# ----- Federated Averaging -----
def average_weights(state_dicts):
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(state_dicts))
    return avg_state_dict

# ----- Testing -----
def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100. * correct / total

# ----- Main Loop -----
def federated_learning(model_type, rounds, delta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_datasets, test_loader = load_data(delta)

    global_model = get_model(model_type).to(device)

    results = []
    for round in tqdm(range(rounds), desc="Federated Learning Rounds"):
        local_weights = []
        for i, dataset in enumerate(client_datasets):
            local_model = copy.deepcopy(global_model)
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            local_state_dict = train_local(local_model, train_loader, device)
            local_weights.append(local_state_dict)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        accuracy = test(global_model, test_loader, device)
        results.append((round + 1, accuracy))

    filename = f"result/IID_cifar10_{model_type}_{delta}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["round", "accuracy"])
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning with CIFAR-10")
    parser.add_argument("--model", type=str, default="DNN", choices=["LR", "DNN", "CNN", "DenseNet"], help="Model type")
    parser.add_argument("--rounds", type=int, default=100, help="Number of federated learning rounds")
    parser.add_argument("--delta", type=float, default=0.0, help="Delta value controlling shared data proportion (0.0 to 1.0)")
    args = parser.parse_args()

    federated_learning(args.model, args.rounds, args.delta)
