#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[5]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[6]:


bos = pd.DataFrame(boston.data)


# In[ ]:





# In[7]:


bos.info()


# In[8]:


bos.head()


# In[9]:


bos.keys()


# In[10]:


boston.keys()


# In[11]:


type(boston.keys())


# In[12]:


boston.data


# In[13]:


boston.feature_names


# In[14]:


bos.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
bos.head()


# In[15]:


boston.target


# In[16]:


bos['MEDV'] = boston.target


# In[17]:


bos.info()


# In[18]:


bos


# In[19]:


type(bos)


# In[21]:


bos.drop('INDUS', axis=1)


# In[23]:


bos=bos.drop('RAD', axis=1)


# In[24]:


bos


# In[28]:


bos = bos.drop(['INDUS','NOX','AGE','B','CHAS'], axis=1)


# In[26]:


bos


# In[29]:


bos = bos.drop(['CHAS','ZN'], axis=1)


# In[30]:


bos


# In[31]:


X = bos.drop('MEDV', axis = 1)


# In[32]:


Y=bos['MEDV']


# In[33]:


X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)


# In[34]:


from sklearn import  cross_validation;


# In[35]:


from sklearn.model_selection import cross_validate


# In[36]:


X_train, X_test, Y_train, Y_test = cross_validate.train_test_split(X, Y, test_size = 0.33, random_state = 5)


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)


# In[39]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[40]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")


# In[41]:


mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


# In[ ]:




