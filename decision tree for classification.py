#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np


# In[56]:


pwd


# In[57]:


dataset=pd.read_csv("mallcustomerdiscount.csv.csv")
dataset


# In[58]:


dataset


# In[59]:


X= dataset.iloc[:,:-1]


# In[60]:


X


# In[61]:


y= dataset.iloc[:,5]


# In[62]:


y


# In[63]:


from sklearn.preprocessing import LabelEncoder


# In[66]:


labelencoder_X = LabelEncoder()


# In[67]:


X= X.apply(LabelEncoder().fit_transform)


# In[68]:


X


# In[69]:


from sklearn.tree import DecisionTreeClassifier


# In[70]:


regressor = DecisionTreeClassifier()


# In[71]:


regressor.fit=(X.iloc[:,1:5], y)


# In[72]:


X_in = np.array([1,1,0,0])


# In[52]:


y_pred = regressor.predict([X_in])


# In[26]:


y_pred


# In[27]:


from sklearn.externals.six import StringIO


# In[28]:


from IPython.display import Image


# In[29]:


from sklearn. tree import export_graphviz


# In[30]:


import pydotplus


# In[31]:


dot_data= StringIO()


# In[32]:


export_graphviz(regressor, out_file= dot_data, filled = True, rounded = Ture, special_characters = True)


# In[73]:


graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.white_png('tree.png')


# In[ ]:




