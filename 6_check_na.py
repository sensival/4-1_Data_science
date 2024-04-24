import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")
df = pd.DataFrame(df)
print("원본")
print(df)

# nan 값이 얼마나 있는지 column별로 확인하기
print("Nan확인1")
print(df.isnull().sum())


# 전체 data 개수 대비 NaN의 비율
print("Nan확인2")
df.isnull().sum() / len(df)

# 결측치 row 날려버리기
# 튜플에서 데이터가 하나라도 없으면 날려버리기
print("Nan하나라도 있으면 행 날리기")
df1 = df.dropna()
print(df1)

# 모든 데이터가 NaN일 때만 날려버리기
print("모든 데이터가 nan일때 행 날리기")
df2 = df.dropna(how='all')
print(df2)

# column을 기준으로 nan 값이 4개 이상이면 해당 column  날려버리기
print("Nan이 4개 이상이면 열 날리기")
df3 = df.dropna(axis=1, thresh=4)
print(df3)

# 결측치를 채워버리기
print("Nan을 0으로 채우기")
df4 =  df['age'].fillna(0)
print(df4)

# 평균값으로 채워버리기
print("Nan을 평균으로 채우기")
df5 = df['age'].fillna(df['age'].mean(), inplace=True)                                    
print(df5)

# 그룹 범주로 나눠서 그룹별 평균값으로 채워버리기
print("Nan을 같은 성별 평균으로 채우기")
df6 = df['age'].fillna(df.groupby('sex')['age'].transform('mean'), inplace=True)
print(df6)

# 컬럼 A와 B 모두 Null이 아닌 경우만 표시
print("성별과 나이가  둘다 Nan이 아닌경우")
df7=df[df['age'].notnull() & df['deck'].notnull()] 
print(df7)
