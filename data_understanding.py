
# coding: utf-8

# In[20]:


#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv("data/HR_comma_sep.csv")


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.sample(5)


# In[6]:


# print unique values of features
for col in df:
    print col + " unique values:"
    print df[col].unique()


# In[7]:


df.columns


# In[8]:


# divide the features in numerical and categorical
numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
categorical_features = ['Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']
numerical_df = df[numerical_features]
categorical_df = df[categorical_features]


# In[9]:


numerical_df.sample(5)


# In[10]:


categorical_df.sample(5)


# In[11]:


# plot histograms for numerical features and bar charts for categorical features

# Set up a grid of plots
fig = plt.figure(figsize=(10, 10)) 
fig_dims = (3, 2)

positions = itertools.product(range(3), range(2))

# plot bar charts
for col, pos in zip(categorical_df, positions) :
    plt.subplot2grid(fig_dims, pos)
    categorical_df[col].value_counts().plot(kind='bar', title= col + ' Counts')
    #plt.xticks(rotation=0) 

fig


# In[12]:


fig = plt.figure(figsize=(10, 10)) 
positions = itertools.product(range(3), range(2))

# plot histograms -- Sturge's rule for the number of bins
for col, pos in zip(numerical_df, positions):
    plt.subplot2grid(fig_dims, pos)
    numerical_df[col].hist(bins=(np.ceil(np.log2(len(numerical_df[col])) + 1)))
    plt.ylabel('Frequency')
    plt.xlabel(col)

fig


# ### transform categorical features to numerical features

# In[13]:


sales_unique = sorted(df['sales'].unique())
salary_unique = sorted(df['salary'].unique())
sales_mapping = dict(zip(sales_unique, range(0, len(sales_unique) + 1)))
salary_mapping = dict(zip(salary_unique, range(0, len(salary_unique) + 1)))
df['sales'].replace(sales_mapping, inplace=True)
df['salary'].replace(salary_mapping, inplace=True)
df.sample(5)


# # Outlier detection

# In[14]:


# scale features between 0 and 1
scaler = MinMaxScaler()
df_01_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_01_scaled.head()


# In[26]:


# box plots
# dictionary key = boxes, medians, whiskers, caps, fliers 
#box_plot_dict = plt.boxplot([row for row in X_scaled], labels=[df.columns])
df_01_scaled.boxplot(figsize=(20,10))

