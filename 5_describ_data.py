import numpy as np 

x = np.array([ 18,   5,  10,  23,  19,  -8,  10,   0,   0,   5,   2,  15,   8,
                2,   5,   4,  15,  -1,   4,  -7, -24,   7,   9,  -6,  23, -13,
                1,   0,  16,  15,   2,   4,  -7, -18,  -2,   2,  13,  13,  -2,
               -2,  -9, -13, -16,  20,  -4,  -3, -11,   8, -15,  -1,  -7,   4,
               -4, -10,   0,   5,   1,   4,  -5,  -2,  -5,  -2,  -7, -16,   2,
               -3, -15,   5,  -8,   1,   8,   2,  12, -11,   5,  -5,  -7,  -4])


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


import scipy.stats
scipy.stats.describe(x)



import pandas as pd
s = pd.Series(x)
s.describe()



import collections 
num_friends = [100,40,30,30,30,30,30,30,30,30,54,54,54,54,54,54,54,54,25,3,100,100,100,3,3] 
# collections.Counter 빈도 수 요약하기
friend_counts = collections.Counter(num_friends) # key와 value로 리턴
print('friends:', friend_counts)


import matplotlib.pyplot as plt 


friend_counts = collections.Counter(num_friends) 
print('friends:', friend_counts) 

# 가시화 추가 
xs = range(101) 
ys = [friend_counts[x] for x in xs]  # key에 대한 value가 없는 경우 0 return

plt.bar(xs,ys) 
plt.axis([0,101,0,25]) 
plt.xlabel("# of friends") 
plt.ylabel("# of people") 
plt.show()




def mean(x) : 
    return sum(x) / len(x) 

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

