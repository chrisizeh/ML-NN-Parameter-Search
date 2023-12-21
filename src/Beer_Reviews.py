#!/usr/bin/env python
# coding: utf-8

# # Preprocessing + NN Playing for Beer Reviews

# In[2]:


# In[4]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import arff


# ## Load Dataset and create Dataframe

# In[5]:


data = arff.load(open('../data/beer_reviews.arff', 'r'))
attr = np.array(data['attributes'])
numericals = [i[0] for i in attr if i[1] == 'INTEGER' or i[1] == 'REAL']
df = pd.DataFrame(data['data'], columns=attr[:, 0])
df.columns


# Use only 10% of the data

# In[6]:


# from sklearn.model_selection import train_test_split
# _, df = train_test_split(df, test_size=0.1, random_state=42)


# In[7]:


len(df)


# In[8]:


df.columns

df['review_time'] = df['review_time'].apply(lambda sec: pd.Timestamp(sec, unit='s'))
print(df.head())


# ## Fix Missing Values
# 
# The rows with missing brewery name for id 1193 are found through a quick google search and added. For the ones with brewery id 27 where the beers already exist with the correct brewery, so I add it based on the dataset. For the others I google with the provided data.
# 
# The missing review profilenames are set to anonynoums, but otherwise kept, because the review is still done correctly.

# In[9]:


print(df[df.isna().any(axis=1)])
print(df[df['brewery_name'].isna()])

print(df[df['brewery_id'] == 1193])
df.loc[df['brewery_id'] == 1193, 'brewery_name'] = 'Crailsheimer Engel-Bräu'
df.loc[df['brewery_id'] == 1193, 'beer_name'] = df.loc[df['brewery_id'] == 1193, 'beer_name'].apply(lambda name: name.split(' WRONG')[0])
print(df[df['brewery_id'] == 1193])


# In[10]:


print(df[df['brewery_id'] == 27])

if (1391053 in df.index):
    df.loc[1391053, 'brewery_id'] = 24831
    df.loc[1391053, 'brewery_name'] = 'American Brewing Company'

if (1391051 in df.index):
    df.loc[1391051, 'brewery_id'] = 24831
    df.loc[1391051, 'brewery_name'] = 'American Brewing Company'

if (1391052 in df.index):  
    df.loc[1391052, 'brewery_id'] = 24831
    df.loc[1391052, 'brewery_name'] = 'American Brewing Company'

if (1391049 in df.index):  
    df.loc[1391049, 'brewery_id'] = 782
    df.loc[1391049, 'brewery_name'] = 'City Brewing Company, LLC'
    df.loc[1391049, 'beer_name'] = 'Side Pocket High Gravity Ale'

if (1391050 in df.index):  
    df.loc[1391050, 'brewery_id'] = 782
    df.loc[1391050, 'brewery_name'] = 'City Brewing Company, LLC'
    df.loc[1391050, 'beer_name'] = 'Side Pocket High Gravity Ale'

if (1391043 in df.index):  
    df.drop(1391043, inplace=True)


# In[11]:


df.loc[df['review_profilename'].isna(), 'review_profilename'] = 'Anonymous'


# In[12]:


len(df['beer_style'].unique())


# In[211]:


print(len(df.loc[df['beer_abv'].isna(), 'beer_name'].unique()))

def create_mean(df):
    means = {}
    for style in df['beer_style'].unique():
        mean_abv = df.loc[df['beer_style'] == style, 'beer_abv'].mean()
        means[style] = round(mean_abv, 1)

    return means

def fill_mean(means, row):
    return 

means = create_mean(df)
print(means)
df.loc[df['beer_abv'].isna(), 'beer_abv'] = df.loc[df['beer_abv'].isna()].apply(lambda row: means[row['beer_style']], axis=1)


# In[216]:


print(df[df.isna().any(axis=1)].count())


# ## Bag of Word for Brewery and Beer Name

# In[15]:


from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import sklearn
sklearn.set_config(transform_output="pandas")


# In[16]:


PUNCT_TO_REMOVE = punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# In[17]:


df["brewery_name"] = df["brewery_name"].str.lower()
df["brewery_name"] = df["brewery_name"].apply(lambda text: remove_punctuation(text))

df["beer_name"] = df["beer_name"].str.lower()
df["beer_name"] = df["beer_name"].apply(lambda text: remove_punctuation(text))

df["text"] = df["brewery_name"] + " " + df["beer_name"]
df['text']


# In[18]:


cnt = Counter()
for text in df["text"].values:
    for word in text.split():
        cnt[word] += 1
        
FREQWORDS = set([w for (w, wc) in cnt.most_common(20)])

counts = list(cnt.values())
print(len(counts))
print(FREQWORDS)


# In[19]:


n_rare = sum([i <= 100 for i in counts])
print(n_rare)
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare-1:-1]])

def remove_rarewords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

df["text"] = df["text"].apply(lambda text: remove_rarewords(text))


# In[20]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
words.set_index(df.index, inplace=True)
words


# In[21]:


df_bag = df.merge(words, how='left', left_index=True, right_index=True)


# In[22]:


words.index.equals(df.index)


# ## Splitting Training and Test Set

# In[23]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[24]:


le = LabelEncoder()
le.fit(df['beer_style'])
df['class'] = le.transform(df['beer_style'])


# In[26]:


X = df.copy()
X.drop('brewery_name', axis=1, inplace=True)
X.drop('brewery_id', axis=1, inplace=True)
X.drop('beer_name', axis=1, inplace=True)
X.drop('beer_beerid', axis=1, inplace=True)
X.drop('beer_style', axis=1, inplace=True)
X.drop('review_profilename', axis=1, inplace=True)
X.drop('review_time', axis=1, inplace=True)
X.drop('text', axis=1, inplace=True)

y = df['class']
X.drop('class', axis=1, inplace=True)
# X = normalize(X, norm='l2')
X


# In[27]:


X.index.equals(words.index)


# In[28]:


X_train, X_valid, words_train, words_valid, y_train, y_valid = train_test_split(X, words, y, test_size=0.33, random_state=42)


# In[29]:


print(len(X_train))
print(len(words_train))
print(len(y_train))
print(X_train.index.equals(y_train.index))
print(X_train.index.equals(words_train.index))
print(y.max())


# ## Scaling, Feature Selection, Outlier

# ## Check for Outliers

# In[30]:


X_train.hist(bins=30, figsize=(15, 10))


# In[31]:


len(X_train[X_train['beer_abv'] > 31])
len(X_train[X_train['beer_abv'] > 50])
X_train[X_train['beer_abv'] > 50]


# Looks like there exist really that strong beers.

# ## Feature Selection

# In[32]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[33]:


clf = RandomForestClassifier(n_estimators=20, max_features=100).fit(words_train, y_train)
model = SelectFromModel(clf, prefit=True)
words_new = model.transform(words_train)
words_new


# In[34]:


# words_new = words_train[words_train.columns[:100]]


# In[35]:


words_train.index.equals(X_train.index)


# In[36]:


f_i = list(zip(clf.feature_names_in_,clf.feature_importances_))
f_i.sort(key = lambda x : x[1])
f_i = f_i[:20]
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

plt.show()


# In[37]:


X_train_bag = X_train.merge(words_new, how='inner', left_index=True, right_index=True)


# In[38]:


X_train_bag


# In[45]:


X_valid_bag = X_valid.merge(words_valid[words_new.columns], how='inner', left_index=True, right_index=True)
X_valid_bag


# # Find Solution for NN

# In[40]:


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# In[41]:


X_ttrain, X_test, y_ttrain, y_test = train_test_split(X_train_bag.values, y_train.values, test_size=0.3, random_state=42)


# In[42]:


print(X_ttrain)
print(y_ttrain)
print(y_ttrain.max())


# ## Build torch dataset

# In[43]:


assert not np.any(np.isnan(X_ttrain))
assert not np.any(np.isnan(y_ttrain))
assert not np.any(np.isnan(X_test))
assert not np.any(np.isnan(y_test))


# In[46]:


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


# In[47]:


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

# In[ ]:


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(X_ttrain[0]), 250),
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

# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)


# In[ ]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (XX, yy) in enumerate(dataloader):
        XX, yy = XX.to(device), yy.to(device)

        # Compute prediction error
        pred = model(XX)
        loss = loss_fn(pred, yy)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(XX)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# In[ ]:


from sklearn.metrics import f1_score, classification_report

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
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


# In[ ]:


epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


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


# Result for validation set: 90.2%

# # Parameter Search

# Parameter to test: Learning Rate, Batch Size, Layer Nodes, Activation Function, Dropout \
# Activation Function: relu, sigmoid, linear

# ## Shrink Dataset

# Use only 2% of the data for parameter testing.

# In[62]:


_, X_train_bag_small, _, y_train_small = train_test_split(X_train_bag, y_train, test_size=0.01, random_state=42)


# In[63]:


print(len(X_train_bag_small))
print(len(y_train_small))


# In[64]:


X_ttrain_small, X_test_small, y_ttrain_small, y_test_small = train_test_split(X_train_bag_small.values, y_train_small.values, test_size=0.3, random_state=42)


# In[65]:


X_train_small_tensor = X_to_tensor(X_ttrain_small)
y_train_small_tensor = y_to_tensor(y_ttrain_small)

X_test_small_tensor = X_to_tensor(X_test_small)
y_test_small_tensor = y_to_tensor(y_test_small)

train_small_ds = TensorDataset(X_train_small_tensor, y_train_small_tensor)
test_small_ds = TensorDataset(X_test_small_tensor, y_test_small_tensor)


# ## Run Searches

# In[119]:
# In[105]:


def acc_func(loc_pred, loc_y):
    # return (loc_pred.argmax(1) == loc_y).type(torch.float).sum().item()
    return f1_score(loc_y, loc_pred.argmax(1), average='weighted')


# In[121]:


from NNModel import NNModel


# In[107]:


layer = [len(X_ttrain[0]), 250, 164, 164, 104]
nnmodel = NNModel(layer, device, acc_func=acc_func, loss_func=nn.CrossEntropyLoss)


# ## Grid Search

# In[ ]:


test_layer = [[len(X_ttrain[0]), 250, 164, 164, 104], [len(X_ttrain[0]), 25, 16, 16, 104], [len(X_ttrain[0]), 250, 164, 104]]
dict_param_1 = {"learning_rate": [0.001, 0.01, 0.05], "batch_size": [32, 64, 128]}
best, acc = nnmodel.grid_search(dict_param_1, train_small_ds, test_small_ds, epochs=30)
print(best)


# In[93]:


nnmodel.defaults["learning_rate"] = best["learning_rate"]
nnmodel.defaults["batch_size"] = best["batch_size"]
dict_param_2 = {"activation": [nn.ReLU, nn.Sigmoid, nn.Identity], "dropout": [0, 0.2, 0.3, 0.5], "layer": test_layer}
best, acc = nnmodel.grid_search(dict_param_2, train_small_ds, test_small_ds, epochs=30)
print(best)


# In[86]:


nnmodel.defaults["activation"] = best["activation"]
nnmodel.defaults["dropout"] = best["dropout"]
nnmodel.defaults["layer"] = best["layer"]


# In[ ]:


print(nnmodel.defaults)
acc = nnmodel.run(nnmodel.defaults, train_ds, test_ds, 30, out=True)
print(acc)


# In[235]:


acc = test(valid_dataloader, nnmodel.model, nnmodel.loss_fn)
print(acc)
validate(valid_dataloader, nnmodel.model)


# ## Local Search

# In[ ]:


nnmodel = NNModel(layer, device, acc_func=acc_func, loss_func=nn.CrossEntropyLoss)


# In[ ]:


init_param = {"learning_rate": 0.01, "batch_size": 64}
best, acc = nnmodel.local_search(init_param, train_small_ds, test_small_ds, steps=50, epochs=30)
print(best)


# In[ ]:


nnmodel.defaults["learning_rate"] = best["learning_rate"]
nnmodel.defaults["batch_size"] = best["batch_size"]
init_param = {"layer": layer, "dropout": 0.2}
best, acc = nnmodel.local_search(init_param, train_small_ds, test_small_ds, steps=50, epochs=30)


# In[ ]:


nnmodel.defaults["dropout"] = best["dropout"]
nnmodel.defaults["layer"] = best["layer"]


# In[ ]:


print(nnmodel.defaults)
acc = nnmodel.run(nnmodel.defaults, train_ds, test_ds, 30, out=True)
print(acc)


# In[ ]:


acc = test(valid_dataloader, nnmodel.model, nnmodel.loss_fn)
print(acc)
validate(valid_dataloader, nnmodel.model)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20, max_features=100, random_state=42)  

rf.fit(X_train, y_train)
y_prediction = rf.predict(X_valid)


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score

accuracy = accuracy_score(y_valid, y_prediction)
print(f'Accuracy: {accuracy}')

f1 = f1_score(y_valid, y_prediction, average='weighted')
print(f'F1-Score: {f1}')

