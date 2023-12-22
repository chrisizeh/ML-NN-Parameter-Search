#!/usr/bin/env python
# coding: utf-8

# # Preprocessing + NN Playing for Beer Reviews

# In[56]:
# In[58]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import arff


# ## Load Dataset and create Dataframe

# In[59]:


data = arff.load(open('../data/beer_reviews_small.arff', 'r'))
attr = np.array(data['attributes'])
numericals = [i[0] for i in attr if i[1] == 'INTEGER' or i[1] == 'REAL']
df = pd.DataFrame(data['data'], columns=attr[:, 0])
df.columns


# Use only 10% of the data

# In[60]:


# from sklearn.model_selection import train_test_split
# _, df = train_test_split(df, test_size=0.1, random_state=42)


# In[61]:


len(df)


# In[62]:


df.columns

df['review_time'] = df['review_time'].apply(lambda sec: pd.Timestamp(sec, unit='s'))
print(df.head())


# ## Fix Missing Values
# 
# The rows with missing brewery name for id 1193 are found through a quick google search and added. For the ones with brewery id 27 where the beers already exist with the correct brewery, so I add it based on the dataset. For the others I google with the provided data.
# 
# The missing review profilenames are set to anonynoums, but otherwise kept, because the review is still done correctly.

# In[63]:


print(df[df.isna().any(axis=1)])
print(df[df['brewery_name'].isna()])

print(df[df['brewery_id'] == 1193])
df.loc[df['brewery_id'] == 1193, 'brewery_name'] = 'Crailsheimer Engel-Bräu'
df.loc[df['brewery_id'] == 1193, 'beer_name'] = df.loc[df['brewery_id'] == 1193, 'beer_name'].apply(lambda name: name.split(' WRONG')[0])
print(df[df['brewery_id'] == 1193])


# In[64]:


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


# In[65]:


df.loc[df['review_profilename'].isna(), 'review_profilename'] = 'Anonymous'


# In[66]:


len(df['beer_style'].unique())


# In[67]:


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


# In[68]:


print(df[df.isna().any(axis=1)].count())


# ## Bag of Word for Brewery and Beer Name

# In[69]:


from string import punctuation
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import sklearn
sklearn.set_config(transform_output="pandas")


# In[70]:


PUNCT_TO_REMOVE = punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# In[71]:


df["brewery_name"] = df["brewery_name"].str.lower()
df["brewery_name"] = df["brewery_name"].apply(lambda text: remove_punctuation(text))

df["beer_name"] = df["beer_name"].str.lower()
df["beer_name"] = df["beer_name"].apply(lambda text: remove_punctuation(text))

df["text"] = df["brewery_name"] + " " + df["beer_name"]
df['text']


# In[72]:


cnt = Counter()
for text in df["text"].values:
    for word in text.split():
        cnt[word] += 1
        
FREQWORDS = set([w for (w, wc) in cnt.most_common(20)])

counts = list(cnt.values())
print(len(counts))
print(FREQWORDS)


# In[73]:


n_rare = sum([i <= 100 for i in counts])
print(n_rare)
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare-1:-1]])

def remove_rarewords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

df["text"] = df["text"].apply(lambda text: remove_rarewords(text))


# In[74]:


del RAREWORDS
del FREQWORDS


# In[75]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
words.set_index(df.index, inplace=True)
words


# In[76]:


# df_bag = df.merge(words, how='left', left_index=True, right_index=True)


# In[77]:


words.index.equals(df.index)


# ## Splitting Training and Test Set

# In[78]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[79]:


le = LabelEncoder()
le.fit(df['beer_style'])
df['class'] = le.transform(df['beer_style'])


# In[80]:


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


# In[81]:


del df


# In[82]:


X.index.equals(words.index)


# In[83]:


# words = words[words.columns[:100]]


# In[84]:


X_train, X_valid, words_train, words_valid, y_train, y_valid = train_test_split(X, words, y, test_size=0.33, random_state=42)


# In[85]:


print(len(X_train))
print(len(words_train))
print(len(y_train))
print(X_train.index.equals(y_train.index))
print(X_train.index.equals(words_train.index))
print(y.max())


# ## Scaling, Feature Selection, Outlier

# ## Check for Outliers

# In[86]:


X_train.hist(bins=30, figsize=(15, 10))


# In[87]:


len(X_train[X_train['beer_abv'] > 31])
len(X_train[X_train['beer_abv'] > 50])
X_train[X_train['beer_abv'] > 50]


# Looks like there exist really that strong beers.

# ## Feature Selection

# In[88]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[89]:


clf = RandomForestClassifier(n_estimators=20, max_features=100).fit(words_train, y_train)
model = SelectFromModel(clf, prefit=True)
words_new = model.transform(words_train)
words_new


# In[90]:


# words_new = words_train[words_train.columns[:100]]


# In[91]:


words_train.index.equals(X_train.index)


# In[92]:


f_i = list(zip(clf.feature_names_in_,clf.feature_importances_))
f_i.sort(key = lambda x : x[1])
f_i = f_i[:20]
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])

plt.show()


# In[93]:


X_train_bag = X_train.merge(words_new, how='inner', left_index=True, right_index=True)


# In[94]:


X_train_bag


# In[95]:


X_valid_bag = X_valid.merge(words_valid[words_new.columns], how='inner', left_index=True, right_index=True)
X_valid_bag


# In[96]:


del words
del words_new
del words_valid
del words_train


# In[ ]:


X_valid_bag.to_csv('../data/beer_target_valid.csv')
y_valid.to_csv('../data/beer_valid.csv')
X_train_bag.to_csv('../data/beer_train.csv')
y_train.to_csv('../data/beer_target_train.csv')

