import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

import random
import time

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

    def __init__(self, layer, device, acc_func, loss_func, reshape=False):
        self.layer = layer
        self.acc_func = acc_func
        self.loss_func = loss_func
        self.reshape = reshape

        self.options = {"activation": [nn.ReLU, nn.Sigmoid, nn.Identity]}
        self.possible_keys = ["learning_rate", "batch_size", "layer", "activation", "dropout"]
        self.defaults = {"learning_rate": 0.01, "batch_size": 64, "layer": self.layer, "activation": nn.ReLU, "dropout": 0}

        self.device = device
        self.create_model(layer)


    def create_model(self, layer, activation=nn.ReLU, dropout=0):
        self.model = NeuralNetwork().to(self.device)

        layer_nn = []
        for i in range(len(layer) - 1):
            layer_nn.append(nn.Linear(layer[i], layer[i+1]))
            layer_nn.append(activation())

            if(dropout > 0):
                layer_nn.append(nn.Dropout(dropout))

        self.model.linear_relu_stack = nn.Sequential(*layer_nn)


    def train(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (XX, yy) in enumerate(dataloader):
            XX, yy = XX.to(self.device), yy.to(self.device)
            if (self.reshape):
                yy = yy.reshape((yy.shape[0], 1))

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
                if (self.reshape):
                    yy = yy.reshape((yy.shape[0], 1))

                pred = self.model(XX)
                test_loss += loss_fn(pred, yy).item()
                correct += self.acc_func(pred, yy)
        test_loss /= num_batches
        correct /= size
        return (100*correct)
    

    def run(self, params, train_ds, test_ds, epochs):
        train_dataloader = DataLoader(train_ds, batch_size=params["batch_size"])
        test_dataloader = DataLoader(test_ds, batch_size=params["batch_size"])

        self.create_model(params["layer"], params["activation"], params["dropout"])
        self.loss_fn = self.loss_func()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params["learning_rate"])
        
        for t in range(epochs):
            self.train(train_dataloader, self.loss_fn, self.optimizer)
            acc = self.test(test_dataloader, self.loss_fn)
        return acc
    

    def get_all_params(self, test_params, keys):
        params = {}
        for key in self.possible_keys:
            if(key in keys):
                params[key] = test_params[keys.index(key)]
            else:
                params[key] = self.defaults[key]
        return params


    def grid_search(self, dict_param, train_ds, test_ds, epochs=2):
        start = time.time()
        best_param = {}
        best_acc = -1

        iter_params = product(*dict_param.values())
        keys = list(dict_param.keys())
        for param_comb in iter_params:
            params = self.get_all_params(param_comb, keys)
            acc = self.run(params, train_ds, test_ds, epochs)
            if(acc > best_acc):
                best_param = param_comb
                best_acc = acc
            print(f"Parameter Combination {str(param_comb)} with keys {str(keys)}\n Accuracy: {acc:>0.1f}\n")
        end = time.time()
        print(f"Grid search took {round((end - start)/60, 1)} minutes.")
        return best_param, best_acc


    def neighborhood(self, params, keys):
        param_list = {}
        for i, val in enumerate(params):
            if(type(val) == float):
                param_list[keys[i]] = [round(val * random.uniform(0.7, 0.9), 5), round(val * random.uniform(1.1, 1.3), 5)]
            elif(type(val) == int):
                param_list[keys[i]] = [int(val * random.uniform(0.7, 0.9)), int(val * random.uniform(1.1, 1.3))]
            elif(type(val) == list):
                param_list[keys[i]] = [[val[0]], [val[0]]]
                for j in range(1, len(val)-1):
                    param_list[keys[i]][0].append(int(val[j] * random.uniform(0.7, 0.9)))
                    param_list[keys[i]][1].append(int(val[j] * random.uniform(1.1, 1.3)))

                param_list[keys[i]][0].append(val[-1])
                param_list[keys[i]][1].append(val[-1])
            elif(callable(val)):
                param_list[keys[i]] = self.options[keys[i]]
        return param_list


    def local_search(self, init_param, train_ds, test_ds, epochs=2, steps=50): 
        start = time.time()
        keys = list(init_param.keys())
        params = self.get_all_params(list(init_param.values()), keys)
        best_acc = self.run(params, train_ds, test_ds, epochs)
        
        best_param = list(init_param.values())

        for i in range(steps):
            print(f"Step {i}")
            print(f"Best Params, Parameter Combination {str(best_param)} with keys {str(keys)}\n Accuracy: {best_acc:>0.1f}\n")
            neighbor_params = self.neighborhood(best_param, keys)
            params, acc = self.grid_search(neighbor_params, train_ds, test_ds, epochs=epochs)

            if(acc > best_acc):
                best_acc = acc
                best_param = params

        end = time.time()
        print(f"Local search took {round((end - start)/60, 1)} minutes.")
        return best_param, best_acc