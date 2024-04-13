
from io import StringIO
import pandas as pd

csv_data = StringIO("""
x1,x2,x3,x4,x5
1,0.1,"1",2019-01-01,A
2,,,2019-01-02,B
3,,"3",2019-01-03,C
,0.4,"4",2019-01-04,A
5,0.5,"5",2019-01-05,B
,,,2019-01-06,C
7,0.7,"7",,A
8,0.8,"8",2019-01-08,B
9,0.9,,2019-01-09,C
""")

df = pd.read_csv(csv_data, dtype={"x1": pd.Int64Dtype()}, parse_dates=[3])
df.dtypes

# In[6]:


df

# In[7]:


df.isnull()

# In[8]:


df.isnull().sum()

# In[9]:


import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)
plt.show()


# In[10]:


msno.bar(df)
plt.show()

# In[11]:


df.dropna()


# In[12]:


df.dropna(thresh=7, axis=1)

# In[13]:


df.dropna(thresh=7, axis=1)

# In[14]:


df.dropna(thresh=4)

# In[15]:


df

# In[16]:


import seaborn as sns 
df = sns.load_dataset("titanic")
df.tail()


# In[17]:


msno.bar(df)
plt.show()

# In[18]:



sns.distplot(df.age.dropna())
plt.show()

# In[19]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df_copy1 = df.copy()
df_copy1["age"] = imputer.fit_transform(df.age.values.reshape(-1,1))

msno.bar(df_copy1)
plt.show()

# In[ ]:


df.groupby(df.pclass).age.median()
df_copy2 = df.copy()
df_copy2["age"] = df.groupby(df.pclass).age.transform(lambda x: x.fillna(x.median()))

g = sns.FacetGrid(df_copy2, hue="pclass", height=4, aspect=2)
g.map(sns.kdeplot, "age")
plt.show()

