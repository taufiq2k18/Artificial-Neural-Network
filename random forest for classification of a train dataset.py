#!/usr/bin/env python
# coding: utf-8

# In[2]:


pwd


# In[10]:


import pandas as pd
import numpy as np
import matpotlib as mpl
import matpotlib.pylot as plt


# In[7]:


dataset=pd.read_csv("train.csv")


# In[8]:


dataset


# In[11]:


dataset.head(5)


# In[16]:


NAs= pd.concat([dataset.isnull().sum()], axis=1, keys=["dataset"])
NAs[NAs.sum(axis=1)>0]


# In[18]:


dataset["Age"]= dataset["Age"].fillna(dataset["Age"].mean())


# In[19]:


dataset["Embarked"]= dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])


# In[20]:


dataset["Cabin"]= dataset["Cabin"].fillna(dataset["Embarked"].mode()[0])


# In[22]:


dataset["Pclass"]= dataset["Pclass"].apply(str)


# In[30]:


for col in dataset.dtypes[dataset.dtypes=="object"].index:
    for_dummy= dataset.pop(col)
    dataset= pd.concat([dataset,pd.get_dummies(for_dummy,prefix=col)],axis=1)
    dataset.head()


# In[33]:


labels = dataset.pop("Survived")
from sklearn
from sklearn.ensemble import RandomForestClassifier


# In[ ]:




