#!/usr/bin/env python
# coding: utf-8

# # Load Preprocessed Dataset

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


X_valid_bag = pd.read_csv('../data/beer_target_valid.csv')
y_valid = pd.read_csv('../data/beer_valid.csv')
X_train_bag = pd.read_csv('../data/beer_train.csv')
y_train = pd.read_csv('../data/beer_target_train.csv')


# # Find Solution for NN

# In[97]:


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# In[98]:

from sklearn.model_selection import train_test_split

X_ttrain, X_test, y_ttrain, y_test = train_test_split(X_train_bag.values, y_train.values, test_size=0.3, random_state=42)


# In[99]:


print(X_ttrain)
print(y_ttrain)
print(y_ttrain.max())


# ## Build torch dataset

# In[100]:


assert not np.any(np.isnan(X_ttrain))
assert not np.any(np.isnan(y_ttrain))
assert not np.any(np.isnan(X_test))
assert not np.any(np.isnan(y_test))


# In[101]:


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# convert a df to tensor to be used in pytorch
def X_to_tensor(df):
    return torch.from_numpy(df).float().to(device)

def y_to_tensor(df):
    return torch.from_numpy(df).long().to(device)

X_train_tensor = X_to_tensor(X_ttrain)
y_train_tensor = y_to_tensor(y_ttrain)

X_test_tensor = X_to_tensor(X_test)
y_test_tensor = y_to_tensor(y_test)

X_valid_tensor = X_to_tensor(X_valid_bag.values)
y_valid_tensor = y_to_tensor(y_valid.values)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)
valid_ds = TensorDataset(X_valid_tensor, y_valid_tensor)


# In[102]:


del X_ttrain
del X_test

del y_ttrain
del y_test


# In[103]:


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_ds, batch_size=batch_size)
test_dataloader = DataLoader(test_ds, batch_size=batch_size)
valid_dataloader = DataLoader(valid_ds, batch_size=len(valid_ds))

for XX, yy in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {XX.shape}")
    print(f"Shape of y: {yy.shape} {yy.dtype}")
    break


# ## Creating Models

# In[126]:


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(train_ds[0][0]), 250),
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

model = NeuralNetwork().to(device)
print(model)


# ## Train the Model

# In[127]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)


# In[128]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    test_loss = 0
    for batch, (XX, yy) in enumerate(dataloader):
        XX, yy = XX.to(device), yy.to(device)

        # Compute prediction error
        pred = model(XX)
        loss = loss_fn(pred, yy)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        test_loss += loss

        if batch % 100 == 0:
            current = (batch + 1) * len(XX)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return test_loss / num_batches


# In[129]:


from sklearn.metrics import f1_score, classification_report

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for XX, yy in dataloader:
            XX, yy = XX.to(device), yy.to(device)
            pred = model(XX)
            test_loss += loss_fn(pred, yy).item()
            # correct += (pred.argmax(1) == yy).type(torch.float).sum().item()
            correct += f1_score(yy, pred.argmax(1), average='weighted')
    test_loss /= num_batches
    correct /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


# In[130]:


losses = []
test_losses = []
accs = []

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    losses.append(train(train_dataloader, model, loss_fn, optimizer))
    test_loss, acc = test(test_dataloader, model, loss_fn)

    accs.append(acc)
    test_losses.append(test_loss)
print("Done!")


# In[ ]:


sns.set()

plt.plot(range(len(accs)), accs)
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.savefig(f"../results/beer_init_nn_acc.png", bbox_inches="tight")
plt.clf()

plt.plot(range(len(losses)), losses, label="Training")
plt.plot(range(len(test_losses)), test_losses, label="Test")
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.savefig(f"../results/beer_init_nn_loss.png", bbox_inches="tight")
plt.legend(loc="upper left")
plt.clf()


# Best: 89.5% -> 30 epochs

# ## Test model

# In[ ]:


def validate(dataloader, model):
    num_batches = len(dataloader)
    assert num_batches == 1
    model.eval()
    with torch.no_grad():
        for XX, yy in dataloader:
            XX, yy = XX.to(device), yy.to(device)
            pred = model(XX)
            print(classification_report(yy, pred.argmax(1)))


# In[ ]:


print(test(valid_dataloader, model, loss_fn))
validate(valid_dataloader, model)


# In[ ]:


del model
del loss_fn
del optimizer

del train_dataloader
del test_dataloader


# Result for validation set: 90.2%

# # Parameter Search

# Parameter to test: Learning Rate, Batch Size, Layer Nodes, Activation Function, Dropout \
# Activation Function: relu, sigmoid, linear

# ## Shrink Dataset

# Use only 2% of the data for parameter testing.

# In[138]:


_, X_train_bag_small, _, y_train_small = train_test_split(X_train_bag, y_train, test_size=0.01, random_state=42)


# In[139]:


print(len(X_train_bag_small))
print(len(y_train_small))


# In[140]:


X_ttrain_small, X_test_small, y_ttrain_small, y_test_small = train_test_split(X_train_bag_small.values, y_train_small.values, test_size=0.3, random_state=42)


# In[141]:


X_train_small_tensor = X_to_tensor(X_ttrain_small)
y_train_small_tensor = y_to_tensor(y_ttrain_small)

X_test_small_tensor = X_to_tensor(X_test_small)
y_test_small_tensor = y_to_tensor(y_test_small)

train_small_ds = TensorDataset(X_train_small_tensor, y_train_small_tensor)
test_small_ds = TensorDataset(X_test_small_tensor, y_test_small_tensor)


# ## Run Searches

# In[142]:


# In[144]:


def acc_func(loc_pred, loc_y):
    # return (loc_pred.argmax(1) == loc_y).type(torch.float).sum().item()
    return f1_score(loc_y, loc_pred.argmax(1), average='weighted')


# In[145]:


from NNModel import NNModel


# In[147]:


layer = [len(train_small_ds[0][0]), 250, 164, 164, 104]
nnmodel = NNModel(layer, device, acc_func=acc_func, loss_func=nn.CrossEntropyLoss)


# ## Grid Search

# In[148]:


test_layer = [[len(train_small_ds[0][0]), 250, 164, 164, 104], [len(train_small_ds[0][0]), 25, 16, 16, 104], [len(train_small_ds[0][0]), 250, 164, 104]]
dict_param_1 = {"learning_rate": [0.001, 0.01, 0.05], "batch_size": [320, 640, 1280]}
best, acc = nnmodel.grid_search(dict_param_1, train_small_ds, test_small_ds, epochs=50)
print(best)


# In[ ]:


nnmodel.defaults["learning_rate"] = best["learning_rate"]
nnmodel.defaults["batch_size"] = best["batch_size"]
dict_param_2 = {"activation": [nn.ReLU, nn.Sigmoid, nn.Identity], "dropout": [0, 0.2, 0.3, 0.5], "layer": test_layer}
best, acc = nnmodel.grid_search(dict_param_2, train_small_ds, test_small_ds, epochs=50)
print(best)


# In[ ]:


nnmodel.defaults["activation"] = best["activation"]
nnmodel.defaults["dropout"] = best["dropout"]
nnmodel.defaults["layer"] = best["layer"]


# In[ ]:


print(nnmodel.defaults)
acc = nnmodel.run(nnmodel.defaults, train_ds, test_ds, 100, out=True, name="beer_grid_res")
print(acc)


# In[ ]:


acc = test(valid_dataloader, nnmodel.model, nnmodel.loss_fn)
print(acc)
validate(valid_dataloader, nnmodel.model)


# In[ ]:


grid_best = nnmodel.defaults


# ## Local Search

# In[ ]:


nnmodel = NNModel(layer, device, acc_func=acc_func, loss_func=nn.CrossEntropyLoss)


# In[ ]:


init_param = {"learning_rate": grid_best["learning_rate"], "batch_size": grid_best["batch_size"]}
best, acc = nnmodel.local_search(init_param, train_small_ds, test_small_ds, steps=5, epochs=50)
print(best)


# In[ ]:


nnmodel.defaults["learning_rate"] = best["learning_rate"]
nnmodel.defaults["batch_size"] = best["batch_size"]
init_param = {"layer": grid_best["layer"], "dropout": grid_best["dropout"]}
best, acc = nnmodel.local_search(init_param, train_small_ds, test_small_ds, steps=5, epochs=50)


# In[ ]:


nnmodel.defaults["dropout"] = best["dropout"]
nnmodel.defaults["layer"] = best["layer"]


# In[ ]:


print(nnmodel.defaults)
acc = nnmodel.run(nnmodel.defaults, train_ds, test_ds, 100, out=True, name="beer_local_res")
print(acc)


# In[ ]:


acc = test(valid_dataloader, nnmodel.model, nnmodel.loss_fn)
print(acc)
validate(valid_dataloader, nnmodel.model)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20, max_features=100, random_state=42)  

rf.fit(X_train_bag, y_train)
y_prediction = rf.predict(X_valid_bag)


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score

accuracy = accuracy_score(y_valid, y_prediction)
print(f'Accuracy: {accuracy}')

f1 = f1_score(y_valid, y_prediction, average='weighted')
print(f'F1-Score: {f1}')

