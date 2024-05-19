

import pandas as pd
import numpy as np
from sklearn import linear_model

rawData = pd.read_csv('c:/Users/wogns/OneDrive/바탕 화면/Data_study/데이터사이언스/company.csv', encoding='CP949', engine ='python')
# 결측치 있는 행은 drop
drop_mssing_data = rawData.dropna()
# 결측치는 0으로 채움
fill_missing_data = rawData.fillna(0)
# 결측치는 각열의 평균값으로
fill_missing_data_fill_mean = rawData.fillna(rawData.mean())

y=drop_mssing_data['유동비율']
X=drop_mssing_data[['ROE','PER','배당수익률']]

# 선형 회귀 모델 초기화
lm = linear_model.LinearRegression()
# 데이터 입력
model = lm.fit(X,y)

#유동비율이 비어있으면서 ROE, PER, 배당수익률은 존재하는 투플을 가져옵니다.
#데이터 프레임은 논리 연산자가 아니라 비트단위 연산자 사용, '조건'을 저장!
tgt_fill=np.isnan(rawData['유동비율']) & ~np.isnan(rawData['ROE'])& ~np.isnan(rawData['PER'])& ~np.isnan(rawData['배당수익률'])

# 위에서 만든 모델로 유동비율 예측 데이터프레임[조건]
fitted = model.predict(rawData[tgt_fill][['ROE','PER','배당수익률']])

# 조건 tgt_fill에 따라 선택된 행들의 인덱스를 반환
row_indexes = rawData[tgt_fill].index

# 복사본
filled_with_regress = pd.DataFrame(rawData, copy=True)

j=0
# 복사본에 유동 비율 na 인 행만 순회하며 선형회귀로 예측한 값 넣기
for i in row_indexes:
    # loc[a, b] a행 b열
    filled_with_regress.loc[i,'유동비율'] = fitted[j]
    j=j+1

# 유동비율은 채웠으니 나머지 na인것은 드랍
drop_mssing_filled_with_regress = filled_with_regress.dropna()

# 여러개의 열을 데이터프레임으로
drop_mssing_filled_with_regress[['Name_eng','유동비율']]

# csv로
drop_mssing_filled_with_regress[['Name_eng','유동비율']].to_csv('company_fill.csv', index=False)


