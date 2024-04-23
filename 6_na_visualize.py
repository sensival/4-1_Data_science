
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
print(df.dtypes)
print(df.isnull())
print(df.isnull().sum())


import missingno as msno # 결측치(missing values)를 시각화하는 파이썬 라이브러리
import matplotlib.pyplot as plt

msno.matrix(df) # 데이터프레임 df의 결측치를 매트릭스 형태로 시각화
plt.show()

msno.bar(df)
plt.show()


df1 = df.dropna()
print("Nan 있는 행 다 버리기")
print(df1)

df2 = df.dropna(thresh=7, axis=1)
print("Nan 7개 있는 열 다 버리기")
print(df2)

print("Nan 4개 있는 열 다 버리기")
df3 = df.dropna(thresh=4)
print(df3)


import seaborn as sns 
df = sns.load_dataset("titanic")
df.tail()

msno.bar(df)
plt.show()

df3["age"] = df.age.dropna()
msno.bar(df3)
plt.show()



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median") #  SimpleImputer를 사용하여 결측치를 대체할 값을 설정
df_copy1 = df.copy()
df_copy1["age"] = imputer.fit_transform(df.age.values.reshape(-1,1)) #  fit_transform() 메서드를 사용하여 실제로 결측치를 대체, reshape(-1,1)에서 -1은 차원 알아서 계산하란뜻

msno.bar(df_copy1)
plt.show()

# 각 승객 클래스(pclass)별로 나이(age)의 중앙값을 계산
df.groupby(df.pclass).age.median()
df_copy2 = df.copy()

# 승객 클래스별 중앙값으로 결측치를 대체하여 'age' 열을 업데이트
df_copy2["age"] = df.groupby(df.pclass).age.transform(lambda x: x.fillna(x.median())) # transform() 메서드는 각 그룹에 대해 함수를 적용하고, 결과를 원래의 인덱스에 맞춰 반환하는 기능

# 승객 클래스(pclass)에 따라서 색상을 구분하여 나이(age)에 대한 밀도 추정 플롯을 생성
g = sns.FacetGrid(df_copy2, hue="pclass", height=4, aspect=2)
g.map(sns.kdeplot, "age")
plt.show()

