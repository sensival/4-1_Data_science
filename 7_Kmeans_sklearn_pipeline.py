from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt


# 아이리스 연습용 데이터 불러오기
iris = datasets.load_iris()
labels = pd.DataFrame(iris.target) # 붓꽃의 종류가 타겟열이라고 이미 정해져 나온 데이터셋임 sns 랑 다름!
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=['Sepal length','Sepal width','Petal length','Petal width']
data = pd.concat([data,labels],axis=1)

feature = data[ ['Sepal length','Sepal width']]

scaler = StandardScaler() # StandardScaler는 데이터의 특성을 평균이 0이고 표준 편차가 1인 표준 정규 분포로 변환하는 데 사용됨
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler,model)# 파이프라인은 여러 단계의 처리를 하나의 묶음으로 구성하여 코드를 간결하고 효율적으로 관리 데이터 훈련전 스케일하고 훈련
pipeline.fit(feature)
predict = pd.DataFrame(pipeline.predict(feature))
predict.columns=['predict']
r = pd.concat([feature,predict],axis=1)

ct = pd.crosstab(data['labels'],r['predict'])
print (ct)

# Sepal length/Sepal width의 분포 히스토그램 
plt.subplot(1,2,1)
plt.hist(data['Sepal length'])
plt.title('Sepal length')

plt.subplot(1,2,2)
plt.hist(data['Sepal width'])
plt.title('Sepal width')
plt.show()


# 클러스터 수에 따른 관성 시각화
ks = range(1,10)
inertias = [] # 관성(inertia) :관성이 작을수록 클러스터의 응집도가 높다고 해석

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(feature)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
