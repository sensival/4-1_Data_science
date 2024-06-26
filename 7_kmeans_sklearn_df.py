import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터프레임 정의
df = pd.DataFrame(columns=('x','y'))

# 데이터프레임에 데이터 추가
df.loc[0] =[7,1]
df.loc[1] =[2,1]
df.loc[2] =[4,2]
df.loc[3] =[9,4]
df.loc[4] =[10,5]
df.loc[5] =[10,6]
df.loc[6] =[11,5]
df.loc[7] =[11,6]
df.loc[8] =[15,3]
df.loc[9] =[15,2]
df.loc[10] =[16,4]
df.loc[11] =[15,1]

# 데이터프레임의 값들을 numpy 배열로 가져오기
data_points = df.values

# K-means 클러스터링 모델 초기화 및 학습
kmeans = KMeans(n_clusters=3).fit(data_points)

# 클러스터링 결과를 데이터프레임에 추가
df['cluster_id'] = kmeans.labels_

# 시각화
sns.lmplot(x='x', y='y', data=df, fit_reg=True, scatter_kws={"s":10}, hue="cluster_id")
plt.title("K-means Clustering Plot")
plt.show()

"""
x, y: 데이터프레임(df)의 열 이름을 지정하여 x축과 y축에 사용할 변수를 선택합니다.

data: 시각화에 사용할 데이터프레임을 지정합니다.

fit_reg: 회귀선을 플롯에 포함할지 여부를 지정합니다. 기본값은 True이며, 회귀선을 포함하려면 True로 설정합니다. 회귀선을 제외하려면 False로 설정합니다.

scatter_kws: 산점도에 대한 추가적인 설정을 지정합니다. 예를 들어, scatter_kws={"s": 150}는 산점도의 점 크기를 150으로 설정합니다.

hue: 데이터프레임의 열 이름을 지정하여 색상을 구분할 변수를 선택합니다. 이를 통해 다른 범주형 변수에 따라 데이터를 색으로 구분할 수 있습니다.

lmplot() 함수는 데이터를 시각화하고 선형 관계를 탐색하는 데 유용한 다양한 기능을 제공합니다.

"""
