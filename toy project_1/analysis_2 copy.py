import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from scipy.stats import kstest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

plt.rcParams['font.family'] = 'Malgun Gothic'  

cancer_data = pd.read_csv("시군구_암종_시기 - 시트1.csv")


factor_data = pd.read_csv("지사건_자료 - 시트1.csv")

# print(factor_data.info())

# 지역사회건강조사 자료가 없는 1999-2003, 2004-2008 행 삭제
cancer_data = cancer_data[~cancer_data['시기'].isin(['1999-2003', '2004-2008'])]

# print(cancer_data)
# print(factor_data)


# 시기와 지역이 같은 데이터 조인하기
merged_data = pd.merge(cancer_data, factor_data, on=['시기', '지역'], how='outer')

# print(merged_data.info())


# x축 레이블을 [시기], [지역] 형식으로 변환
xtick_labels = [f"[{col}]" for col in merged_data.columns]

# # NaN 시각화
xtick_labels = [f"{row['시기']} {row['지역']}" for _, row in merged_data.iterrows()]

# 결측치 시각화
# plt.figure(figsize=(15, 7))
# sns.heatmap(merged_data.isnull().T, yticklabels=False, xticklabels=xtick_labels, cmap="viridis")
# plt.title("결측치 시각화")
# plt.show()

# 조건에 따라 행 삭제
# 시기가 2009-2013이고 지역이 세종특별자치시인 행 삭제.
# 지역이 전국인 모든 행 삭제.
# 시기가 2009-2013이고 지역이 제주특별자치도인 행의 결측치 채우기



merged_data = merged_data[merged_data.isnull().sum(axis=1) < 2]

# 결측치가 있는 열만 선택하여 출력
missing_values = merged_data.isnull().sum()
columns_with_missing = missing_values[missing_values > 0]
print("결측행 drop", columns_with_missing)

merged_data['호지킨 림프종(C81)'] = merged_data['호지킨 림프종(C81)'].fillna(merged_data['호지킨 림프종(C81)'].mean())

missing_values = merged_data.isnull().sum()
columns_with_missing = missing_values[missing_values > 0]
print("결측치 채우기", columns_with_missing)


corr_data = merged_data.iloc[:, 2:]

keep_columns = ['삶의질', '당뇨', '고혈압', '스트레스 인지율', '걷기실천율', '월간 음주율', '현재 흡연률', '저염식선호율', 
                '갑상선(C73)', '폐(C33-C34)', '자궁체부(C54)']

# 제외한 열들을 남기고 나머지 열 drop
corr_data = corr_data[keep_columns]
print(corr_data)


# correlation_matrix = corr_data.corr(method='kendall')

# 데이터프레임의 열 이름 리스트
columns = corr_data.columns

# 빈 데이터프레임 생성
kendall_corr_matrix = pd.DataFrame(index=columns, columns=columns)
p_value_matrix = pd.DataFrame(index=columns, columns=columns)

# 상관계수와 p-value 계산
for col1 in columns:
    for col2 in columns:
        corr, p_value = kendalltau(corr_data[col1], corr_data[col2])
        kendall_corr_matrix.loc[col1, col2] = corr
        p_value_matrix.loc[col1, col2] = p_value

# 상관계수와 p-value 결과 출력
print("Kendall Tau Correlation Matrix:")
print(kendall_corr_matrix)
print("\nP-Value Matrix:")
print(p_value_matrix)



plt.figure(figsize=(20, 20))

plt.subplot(1, 2, 1)
sns.heatmap(kendall_corr_matrix.astype(float), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Kendall Tau Correlation Matrix")

# p-value 히트맵
plt.subplot(1, 2, 2)
sns.heatmap(p_value_matrix.astype(float), annot=True, cmap='viridis', fmt=".2f", vmin=0, vmax=1)
plt.title("P-Value Matrix")

plt.tight_layout()
plt.show()