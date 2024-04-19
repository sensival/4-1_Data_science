######### 정규분포 시각화
import math
from matplotlib import pyplot as plt  

# 정규 분포의 확률 밀도 함수(pdf)를 계산하기 위한 함수
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)  # 2*pi의 제곱근을 계산---> 정규분포의 확률 밀도 함수에 쓰임
    # 정규 분포에 대한 확률 밀도 함수
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma)) #  exp() : e의 거듭제곱 값을 계산

# -5부터 5까지 0.1씩 증가하는 x 값 리스트를 생성
xs = [x / 10.0 for x in range(-50,50)]

# mu=0이고 sigma=1인 경우의 정규 분포의 확률 밀도 함수를 그립니다.
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')

# mu=0이고 sigma=2인 경우의 정규 분포의 확률 밀도 함수를 그립니다.
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')

# mu=0이고 sigma=0.5인 경우의 정규 분포의 확률 밀도 함수를 그립니다.
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')

# mu=-1이고 sigma=1인 경우의 정규 분포의 확률 밀도 함수를 그립니다.
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')

plt.legend()
plt.title('다양한 정규 분포의 확률 밀도 함수')
plt.show()


######### 누적 확률분포 시각화
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



######### 이항 분포와 정규 근사(중심 극한 정리) 시각화
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



######### 이항분포 시각화 p=0.5
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


######### 초기하분포 시각화 p=1/3
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


######### 이항분포 시각화 p=random

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


######## 카이제곱 분포
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


###### 포아송분포
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
