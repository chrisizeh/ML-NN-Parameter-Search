import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(282, 250),
            nn.ReLU(),
            nn.Linear(250, 164),
            nn.ReLU(),
            nn.Linear(164, 164),
            nn.ReLU(),
            nn.Linear(164, 104)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NNModel:

    def __init__(self, layer, device, features, target):
        self.layer = layer
        self.features = features
        self.target = target

        self.device = device
        self.create_model()

    def create_model(self):
        self.model = NeuralNetwork().to(self.device)

        self.model.linear_relu_stack = nn.Sequential(
            nn.Linear(self.features, 250),
            nn.ReLU(),
            nn.Linear(250, 164),
            nn.ReLU(),
            nn.Linear(164, 164),
            nn.ReLU(),
            nn.Linear(164, self.target)
        )

    def train(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (XX, yy) in enumerate(dataloader):
            XX, yy = XX.to(self.device), yy.to(self.device)

            # Compute prediction error
            pred = self.model(XX)
            loss = loss_fn(pred, yy)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(XX)

    def test(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for XX, yy in dataloader:
                XX, yy = XX.to(self.device), yy.to(self.device)
                pred = self.model(XX)
                test_loss += loss_fn(pred, yy).item()
                correct += (pred.argmax(1) == yy).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return (100*correct)
    
    def run(self, train_dataloader, test_dataloader, epochs):
        for t in range(epochs):
            self.train(train_dataloader, self.loss_fn, self.optimizer)
            acc = self.test(test_dataloader, self.loss_fn)
        return acc

    def grid_search(self, dict_param, train_ds, test_ds, epochs=2):
        best_param = {}
        best_acc = -1
        for batch_size in dict_param["batch_size"]:
            for learning_rate in dict_param["learning_rate"]:
                train_dataloader = DataLoader(train_ds, batch_size=int(batch_size))
                test_dataloader = DataLoader(test_ds, batch_size=int(batch_size))
        
                self.create_model()
                self.loss_fn = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

                acc = self.run(train_dataloader, test_dataloader, epochs)
                if(acc > best_acc):
                    best_param["batch_size"] = batch_size
                    best_param["learning_rate"] = learning_rate
                    best_acc = acc
                print(f"Batch Size: {batch_size:.0f}, Learning Rate: {learning_rate:>0.3f} \n Accuracy: {acc:>0.1f}\n")

        return best_param, best_acc


    def neighbor_hood(self, params):
        param_list = {}
        for param, val in params.items():
            param_list[param] = [round(val * random.uniform(0.7, 0.9), 5), round(val * random.uniform(1.1, 1.3), 5)]
        return param_list


    def local_search(self, init_param, train_ds, test_ds, epochs=2, steps=50): 
        train_dataloader = DataLoader(train_ds, batch_size=init_param["batch_size"])
        test_dataloader = DataLoader(test_ds, batch_size=init_param["batch_size"])

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=init_param["learning_rate"])

        best_acc = self.run(train_dataloader, test_dataloader, epochs)
        best_param = init_param

        for i in range(steps):
            print(f"Step {i}")
            print(f"Best Params, Batch Size: {best_param['batch_size']:.0f}, Learning Rate: {best_param['learning_rate']:>0.3f}, Acc: {best_acc:>0.1f}")
            print(best_param)
            neighbor_params = self.neighbor_hood(best_param)
            params, acc = self.grid_search(neighbor_params, train_ds, test_ds, epochs=epochs)

            if(acc > best_acc):
                best_acc = acc
                best_param = params
                best_param["batch_size"] = int(best_param["batch_size"])

        return best_param