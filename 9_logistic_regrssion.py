#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

height = [170, 168, 177, 181 ,172, 171, 169, 175, 174, 178, 170, 167,
          177, 182 ,173, 171, 170, 179, 175, 177, 186, 166, 183, 168]
weight = [70, 66, 73, 77, 74, 73, 69, 79, 77, 80, 74, 68, 71, 76, 78, 
          72, 68, 79, 77, 81, 84, 73, 78, 69]

body = pd.DataFrame(
    {'height': height,
    'weight': weight
    }
)

body.tail()

# In[2]:


import statsmodels.api as sm

reg = sm.OLS.from_formula("height ~ weight", body).fit()
reg.summary()

# In[3]:


import pandas as pd

score = [56, 60, 61, 67, 69, 55, 70, 44, 51, 64, 60, 50, 
         68, 72, 90, 93, 85, 74, 81, 88, 92, 97, 77, 78, 98]
_pass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]

result = pd.DataFrame(
    {"score": score, "_pass": _pass}
)

result.tail()

# In[4]:


# 로지스틱 회귀분석하기

import statsmodels.api as sm

logis = sm.Logit.from_formula('_pass ~ score', result).fit()

logis.summary()

# In[15]:


from sklearn.datasets import load_boston
import matplotlib.pyplot as plt 
%matplotlib inline

boston = load_boston()
model_boston = LinearRegression().fit(boston.data, boston.target)

predictions = model_boston.predict(boston.data)

plt.scatter(boston.target, predictions)
plt.xlabel("price")
plt.ylabel("prediction")
plt.title("co-relation")
plt.show()

# In[16]:


model = sm.OLS(y, X)
result = model.fit()
print(result.summary())

# In[21]:


dfX0 = pd.DataFrame(boston.data, columns=boston.feature_names)
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(boston.target, columns=["MEDV"])

model_boston2 = sm.OLS(dfy, dfX)
result_boston2 = model_boston2.fit()
print(result_boston2.summary())

# In[32]:


import numpy as np 
x_new = dfX0.mean().values
dfx_new = sm.add_constant(pd.DataFrame(np.array(x_new)[:, np.newaxis].T,
                                       columns=boston.feature_names),
                          has_constant="add")
dfx_new

# In[33]:


result_boston2.predict(dfx_new)

# In[34]:


dfy.mean()

# In[35]:


result_boston2.params

# In[36]:


result_boston2.resid.plot()
plt.title("resid vector")
plt.xlabel("data num")
plt.show()

# In[37]:


result_boston2.resid.sum()  #잔차의 합은 0이됨

# In[ ]:



