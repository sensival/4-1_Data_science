

# nan 값이 얼마나 있는지 column별로 확인하기
df.isnull().sum()

# 전체 data 개수 대비 NaN의 비율
df.isnull().sum() / len(df)
# 결측치 row 날려버리기
# 튜플에서 데이터가 하나라도 없으면 날려버리기
df = df.dropna()

# 모든 데이터가 NaN일 때만 날려버리기
df = df.dropna(how='all')

# column을 기준으로 nan 값이 4개 이상이면 해당 column  날려버리기
df = df.dropna(axis=1, thres=3)
# 결측치를 채워버리기
# NaN을 0으로 채워버리기
df.fillna(0)

# 평균값으로 채워버리기
df['col1'].fillna(df['col1'].mean(), inplace=True)                                    

# 그룹 범주로 나눠서 그룹별 평균값으로 채워버리기
df['col1'].fillna(df.groupby('sex')['col1'].transform('mean'), inplace=True)
df[df['A'].notnull() & df['B'].notnull()] # 컬럼 A와 B 모두 Null이 아닌 경우만 표시

# In[1]:

