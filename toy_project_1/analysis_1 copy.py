import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from scipy.stats import kstest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Malgun Gothic'  

# data import
cancer_data = pd.read_csv("시군구_암종_시기 - 시트1.csv")

# 모든암 열 drop
cancer_data = cancer_data.drop(columns=['모든 암(C00-C96)'], errors='ignore')


# 숫자형 열의 평균으로 결측치 채우기
numeric_columns = cancer_data.select_dtypes(include=['float64', 'int64'])
cancer_data[numeric_columns.columns] =numeric_columns.fillna(numeric_columns.mean())


# 암 발생률 열 선택 (시기와 지역 열 제외)
cancer_columns = cancer_data.columns[2:]
scaler = RobustScaler()
cancer_data[numeric_columns.columns] = scaler.fit_transform(cancer_data[numeric_columns.columns])


# 지역별 암 발생률 열의 변동계수 계산
range= cancer_data[cancer_columns].std() / cancer_data[cancer_columns].mean()

range_sorted = range.sort_values(ascending=False)

# 출력

print(range_sorted)

