import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

import random
import time
import copy

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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


    def train(self, dataloader, loss_fn, optimizer, out=False):
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
                if out:
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test(self, dataloader, loss_fn, out=False):
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
        correct /= num_batches
        if out:
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return (100*correct), test_loss
    

    def run(self, params, train_ds, test_ds, epochs, out=False):
        train_dataloader = DataLoader(train_ds, batch_size=params["batch_size"])
        test_dataloader = DataLoader(test_ds, batch_size=params["batch_size"])

        self.create_model(params["layer"], params["activation"], params["dropout"])
        self.loss_fn = self.loss_func()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params["learning_rate"])
        early_stopper = EarlyStopper(patience=3)

        for t in range(epochs):
            self.train(train_dataloader, self.loss_fn, self.optimizer, out=out)
            acc, test_loss = self.test(test_dataloader, self.loss_fn, out=out)
            if early_stopper.early_stop(test_loss):     
                print("Early stopping at epoch:", t)
                break
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

        res = {}
        for i in range(len(keys)):
            res[keys[i]] = best_param[i]
        return res, best_acc


    def neighborhood(self, params):
        param_list = {}
        for key, val in params.items():
            if(type(val) == float):
                param_list[key] = [round(val * random.uniform(0.7, 0.9), 5), round(val * random.uniform(1.1, 1.3), 5)]
            elif(type(val) == int):
                param_list[key] = [int(val * random.uniform(0.7, 0.9)), int(val * random.uniform(1.1, 1.3))]
            elif(type(val) == list):
                param_list[key] = [[val[0]], [val[0]]]
                for j in range(1, len(val)-1):
                    param_list[key][0].append(int(val[j] * random.uniform(0.7, 0.9)))
                    param_list[key][1].append(int(val[j] * random.uniform(1.1, 1.3)))

                param_list[key][0].append(val[-1])
                param_list[key][1].append(val[-1])
            elif(callable(val)):
                param_list[key] = self.options[key]
        return param_list


    def local_search(self, init_param, train_ds, test_ds, epochs=2, steps=50): 
        best_param = copy.deepcopy(init_param)
        start = time.time()
        keys = list(init_param.keys())
        params = self.get_all_params(list(best_param.values()), list(best_param.keys()))
        best_acc = self.run(params, train_ds, test_ds, epochs)
        
        for i in range(steps):
            print(f"Step {i}")
            print(f"Best Params, Parameter Combination {str(best_param)}\n Accuracy: {best_acc:>0.1f}\n")
            neighbor_params = self.neighborhood(best_param)
            params, acc = self.grid_search(neighbor_params, train_ds, test_ds, epochs=epochs)

            if(acc > best_acc):
                best_acc = acc
                best_param = params

        end = time.time()
        print(f"Local search took {round((end - start)/60, 1)} minutes.")
        return best_param, best_acc