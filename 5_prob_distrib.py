#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

x = np.array([ 18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
                2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13,
                1,   0,  16,  15,   2,   4,  -7, -18,  -2,   2,  13,  13,  -2,
               -2,  -9, -13, -16,  20,  -4,  -3, -11,   8, -15,  -1,  -7,   4,
               -4, -10,   0,   5,   1,   4,  -5,  -2,  -5,  -2,  -7, -16,   2,
               -3, -15,   5,  -8,   1,   8,   2,  12, -11,   5,  -5,  -7,  -4])

# In[2]:


len(x)  # 갯수
np.mean(x) # 평균
np.var(x) # 분산
np.std(x)  # 표준 편차
np.max(x)  # 최댓값
np.min(x)  # 최솟값
np.median(x)  # 중앙값
np.percentile(x, 25)  # 1사분위 수
np.percentile(x, 50)  # 2사분위 수
np.percentile(x, 75)  # 3사분위 수

# In[5]:


import scipy.stats
scipy.stats.describe(x)

# In[7]:


import pandas as pd
s = pd.Series(x)
s.describe()

# In[8]:


import collections 

num_firends = [100,40,30,54,25,3,100,100,100,3,3] 
friend_counts = collections.Counter(num_firends) 
print('friends:', friend_counts)

# In[10]:


import collections 
import matplotlib.pyplot as plt 

num_firends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3] 
friend_counts = collections.Counter(num_firends) 
print('friends:', friend_counts) 

# 가시화 추가 
xs = range(101) 
ys = [friend_counts[x] for x in xs] 
# 파이썬에는 이렇게 List구축을 할 수 있습니다. 
#list comprehension 라고 말합니다. 

plt.bar(xs,ys) 
plt.axis([0,101,0,25]) 
plt.xlabel("# of friends") 
plt.ylabel("# of people") 
plt.show()

# In[12]:


num_firends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3] 
num_points = len(num_firends) 
print(num_points) # 25 

max_value = max(num_firends) 
print(max_value) # 100 
min_value = min(num_firends) 
print(min_value) # 3


# In[14]:


num_firends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3] 

sorted_values = sorted(num_firends) # 오름차순으로 정렬된 리스트를 반환한다 
second_smallest_value = sorted_values[1] # 두번째로 작은 값 
second_largest_value = sorted_values[-2] # 두번째로 큰 값 , 파이선에서 -1 은 가장 뒤를 말한다

def mean(x) : return sum(x) / len(x) 

avgOfFriends = mean(num_firends) 
print(avgOfFriends)  # 45.84


# In[19]:


num_firends = [1200,15,10,10,9,4,3,3,2,1] 

def mean(x) : 
    return sum(x) / len(x) 

avgOfFriends = mean(num_firends) 
print(avgOfFriends) 

def median(v) : 
    n = len(v) 
    sorted_v = sorted(v) # 정렬해준뒤에 
    midpoint = n // 2 # // 로 나누면 int형이 됨. / 로 나누면 float 

    if n % 2 == 1: 
        return sorted_v(midpoint) # 리스트가 홀 수면 가운데 값 

    else : 
        lo = midpoint - 1 # 짝수면 가운데의 2개의 값의 평균 
        hi = midpoint 
        return (sorted_v[lo]+sorted_v[hi]) / 2 

medianOfFriends = median(num_firends) 
print(medianOfFriends) 

#결과 평균: 125.7 중앙값: 6.5


# In[20]:


import math
from matplotlib import pyplot as plt
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma))
xs = [x / 10.0 for x in range(-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title('Various Normal pdfs')
plt.show()

# In[21]:


import math
from matplotlib import pyplot as plt
def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
xs = [x / 10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4)
plt.title('Various Normal cdfs')
plt.show()

# In[44]:


import math
import random
from collections import Counter
from matplotlib import pyplot as plt

def bernoulli_trial(p):
    return 1 if random.random() < p else 0
def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))
def make_hist(p, n, num_points):
    data = [binomial(n,p) for _ in range(num_points)]

    # 이항분포의 표본을 막대 그래프로 표현
    histrogram = Counter(data)
    plt.bar([x - 0.4 for x in histrogram.keys()],
            [v / num_points for v in histrogram.values()],
            0.8,
            color='0.75')
    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # 근사된 정규분포를 라인 차트로 표현
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]

    plt.plot(xs,ys)
    
plt.title("Binomial Distribution vs. Normal Approximation")
make_hist(0.5,100,10000)  # 성공확률 0.5, 시행 100번, 수행 10000회
plt.show()

# In[48]:


import numpy as np 
import matplotlib
import scipy as sp 
from scipy.stats import binom
import matplotlib.pyplot as plt
from numpy import arange

p = .5

plt.figure(figsize=(12, 8))

for n in arange(4, 41, 4):
    x = arange(n + 1)
    plt.plot(x, binom(n, p).pmf(x), 'o--', label='(n=' + str(n) + ')')
    
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Binomial Distribution(p = .5)')
plt.grid()
plt.legend()
plt.show()

# In[53]:


import matplotlib.pyplot as plt
from scipy.special import factorial, comb



f = factorial(n) / (factorial(x) * factorial(n - x))
plt.figure(figsize=(12, 8))

p = 1 / 3
x = arange(21)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19, 20])

y = comb(n, x) * p ** x * (1 - p) ** (n - x)

plt.bar(x, y)
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Binomial Distribution(n = 20, p = 1/3)')
plt.grid()
plt.show()

# In[54]:


import numpy as np 
import matplotlib
import scipy as sp 
from scipy.stats import binom
import matplotlib.pyplot as plt
from numpy import arange

plt.figure(figsize=(12, 8))

for p in arange(1, 10) / 10:
    plt.plot(x, binom(n, p).pmf(x), 'o--', label='(p=' + str(p) + ')')
    
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Binomial Distribution(n = 20)')
plt.grid()
plt.legend()
plt.show()

# In[56]:


import scipy.stats as stats
import numpy as np

chi2 = np.linspace(0.5, 50, 100) 

nus = [1, 2, 3, 4, 5, 10, 20, 30]    # 자유도

plt.figure(figsize=(10, 6))          # 플롯 사이즈 지정

for nu in nus:
    plt.plot(chi2, stats.chi2(nu).pdf(chi2), label=r'$\chi^2$(' + str(nu) + ')')    
    # plot 추가       
    
plt.xlabel(r'$\chi^2$')              # x축 레이블 지정
plt.ylabel("y")                      # y축 레이블 지정
plt.grid()                           # 플롯에 격자 보이기
plt.title(r'$\chi^2$ Distribution with scipy.stats')     # 타이틀 표시
plt.legend()                         # 범례 표시
plt.show()                           # 플롯 보이기

# In[58]:


from scipy.stats import poisson
from numpy import arange, exp, power
from scipy.special import factorial

x = arange(41, dtype='f8')   # 큰 숫자를 취급하기 위하여 데이터 타입을 실수로 선언
l = 20
y1 = poisson(l).pmf(x)

plt.figure(figsize=(12, 8))

for l in arange(4, 30, 4):
    plt.plot(x, poisson(l).pmf(x), 'o--', label=r'$(\lambda =$' + str(l) + ')')
    
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Poisson Distribution')
plt.grid()
plt.legend()
plt.show()

# In[ ]:


import sys 
for line in sys.stdin: 
    # remove leading and trailing whitespace 
    line = line.strip() 
    # split the line into words 
    words = line.split() 
    # increase counters 
    for word in words: 
        print ('%s\t%s' % (word, 1))

# In[ ]:


import sys
current_word = None
current_count = 0
word = None
for line in sys.stdin:
    # remove leading and trailing whitespaces
    line = line.strip()
    # parse the input we got from mapper.py
    word, count = line.split('\t', 1)
    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print ('%s\t%s' % (current_word, current_count))
        current_count = count
        current_word = word
if current_word == word:
    print ('%s\t%s' % (current_word, current_count))
