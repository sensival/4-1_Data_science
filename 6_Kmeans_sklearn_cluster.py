from sklearn.cluster import KMeans
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn import datasets
import pandas as pd


# 아이리스 연습용 데이터 불러오기
iris = datasets.load_iris()
labels = pd.DataFrame(iris.target) # 붓꽃의 종류가 타겟열이라고 이미 정해져 나온 데이터셋임 sns 랑 다름!
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data,labels],axis=1)

feature = data[ ['Sepal length','Sepal width']]

# Kmeans 모델 만들기
model = KMeans(n_clusters=3,algorithm='lloyd') # KMeans must be a str among {'elkan:상한지, 하한치로 계산함. 큰데이터에 더 효율적', 'lloyd: 매 반복마다 모든 점들과의 거리를 구해서 중심을 조정함'}.
model.fit(feature) #.fit()   ->   training, 클러스터의 중심 결정

# 예측
predict = pd.DataFrame(model.predict(feature)) #.predict()     -> 예측
predict.columns=['predict']

# concatenate labels to df as a new column
r = pd.concat([feature,predict],axis=1)
print(r)


# 시각화
# model.cluster_centers_: K-means 모델이 클러스터의 중심을 저장하는 속성,numpy 배열
centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal length', 'Sepal width'])

# 중심 좌표를 x, y 변수에 할당
center_x = centers['Sepal length']
center_y = centers['Sepal width']

# scatter plot생성: Sepal length와 Sepal width를 x, y 축으로 사용하며, c=r['predict']는 클러스터링 결과에 따라 색상을 다르게 표시하는 속성
plt.scatter(r['Sepal length'], r['Sepal width'], c=r['predict'], alpha=0.5)

# 클러스터의 중심점은 색깔 다르게
plt.scatter(center_x, center_y, s=50, marker='D', c='r')

# 그래프를 출력
plt.show()


# label과 predict
ct = pd.crosstab(data['labels'],r['predict'])
print (ct)
"""
predict   0   1   2
labels
0         0  50   0
1        12   0  38
2        34   1  15
"""


