#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('bbc_data.csv')


# In[5]:


df


# In[4]:


df['labels'].unique()


# In[6]:


df.isna().sum()


# In[9]:


df_cleaned = df[['data', 'labels']].drop_duplicates()
print(df_cleaned)


# In[10]:


df['labels'].value_counts()


# In[14]:


dir(df)


# In[ ]:




