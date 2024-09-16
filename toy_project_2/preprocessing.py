import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import stats

# 데이터셋 로드
covid_data = pd.read_csv("C:/Users/wogns/OneDrive/바탕 화면/깃허브 레포지토리/Data_study/데이터사이언스/toy_project_2/COVID-19_CBC_Data.csv")

plt.rcParams['font.family'] = 'Malgun Gothic'  


# 결측치 확인
plt.figure(figsize=(10,5))
sns.heatmap(covid_data.isnull(),yticklabels= False)
plt.title("Visualization of Missing Values")
plt.show()

# 열에 쓰일 수 없는 특수문자, 공백 제거
covid_data.columns = covid_data.columns.str.replace(' (Y/N)', '')
covid_data.columns = covid_data.columns.str.replace(' ', '_')
covid_data.columns = covid_data.columns.str.replace('(%)', '')

# 오탈자 수정 및 형식 통일 안된거 수정
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].replace(to_replace=r'O[3-9]', value='O2', regex=True)
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].str.replace(' ', '', regex=False)
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].str.lower().str.replace(r'\s+', '', regex=True)  # 모든 공백 제거


# 'antibiotic', 'antibiotics' 모두를 'antibiotics'로 통합
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].replace('antibiotic', 'antibiotics', regex=False).replace('antibioticss', 'antibiotics', regex=False)

covid_data.to_csv('C:/Users/wogns/OneDrive/바탕 화면/깃허브 레포지토리/Data_study/데이터사이언스/toy_project_2/covid_data_full_2.csv', index=False)


covid_data = pd.read_csv("C:/Users/wogns/OneDrive/바탕 화면/깃허브 레포지토리/Data_study/데이터사이언스/toy_project_2/covid_data_full_3.csv")


# 불필요한 열 (데이터 입력날짜) 열 삭제
covid_data= covid_data.drop(columns=['Sample_Collection_Date_'], axis=1)


# Admission_DATE_와 Discharge_DATE_or_date_of_Death를 datetime 형식으로 변환
covid_data['Admission_DATE_'] = pd.to_datetime(covid_data['Admission_DATE_'])
covid_data['Discharge_DATE_or_date_of_Death'] = pd.to_datetime(covid_data['Discharge_DATE_or_date_of_Death'])

# 재원일수 계산: 퇴원일 - 입원일
covid_data['Hospital_Day'] = (covid_data['Discharge_DATE_or_date_of_Death'] - covid_data['Admission_DATE_']).dt.days

# 결과 확인
print(covid_data[['Admission_DATE_', 'Discharge_DATE_or_date_of_Death', 'Hospital_Day']])

# 재원일수가 음수인 데이터 확인
negative_length_of_stay = covid_data[covid_data['Hospital_Day'] < 0]

# 결과 출력
print(negative_length_of_stay)

# Hospital_Day 양수처리
covid_data['Hospital_Day'] = covid_data['Hospital_Day'].abs()

# Admission_DATE_ Discharge_DATE_or_date_of_Death 삭제
covid_data= covid_data.drop(columns=['Admission_DATE_', 'Discharge_DATE_or_date_of_Death'], axis=1)

# # Outcome 열 이진 인코딩 (Recovered -> 1, Not_recovered -> 0)
covid_data['Outcome'] = covid_data['Outcome'].map({'Recovered': 1, 'Not Recovered': 0})

# Gender 열 이진 인코딩 (Male -> 1, Female -> 0)
covid_data['Gender'] = covid_data['Gender'].map({'Male': 1, 'Female': 0})

# Ventilated_(Y/N) 열 이진 인코딩 (Yes -> 1, No -> 0)
covid_data['Ventilated'] = covid_data['Ventilated'].map({'Yes': 1, 'No': 0})

# 결과 확인
print(covid_data[['Outcome', 'Gender', 'Ventilated']].head())


# one hot 인코딩
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].str.split(',')


mlb = MultiLabelBinarizer()
treatment_encoded = mlb.fit_transform(covid_data['What_kind_of_Treatment_provided_'])

# Create a DataFrame with the one-hot encoded data
treatment_encoded_df = pd.DataFrame(treatment_encoded, columns=mlb.classes_)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
covid_data_encoded = pd.concat([covid_data, treatment_encoded_df], axis=1).drop('What_kind_of_Treatment_provided_', axis=1)

print(covid_data_encoded.head())
covid_data_encoded.to_csv('C:/Users/wogns/OneDrive/바탕 화면/깃허브 레포지토리/Data_study/데이터사이언스/toy_project_2/encoded_data.csv', index=False)


######EDA##########
#Let's check for outliers with #box plot
plt.figure(figsize=(15, 8))
sns.boxplot(data=covid_data_encoded, orient="h")
plt.title("Boxplot of Covid Data")
plt.xlabel("Values")
plt.show()


covid_data.hist(grid=False,
             figsize=(16,16))

plt.show()


# corr Matrix

def calculate_pvalues(data):
    n = data.shape[1]
    pvals = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            _, p = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
            pvals[i, j] = p
            pvals[j, i] = p  # 행렬은 대칭이므로 대칭적으로 채워줍니다.

    return pd.DataFrame(pvals, index=data.columns, columns=data.columns)


correlation_matrix = covid_data_encoded.corr()
pvalue_matrix = calculate_pvalues(covid_data_encoded)

plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# with annot=true parameter, numerical values ​​are added to the cells of the matrix.
# cmap parameter determines the color gradation of the correlation coefficients
plt.title("Correlation Matrix Between Variables")
plt.show()

plt.figure(figsize=(20, 20))
sns.heatmap(pvalue_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.title("P-value Matrix Between Variables")
plt.show()