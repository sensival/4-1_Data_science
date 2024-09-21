# COVID-19 예후 예측 모델

2024.09 - 2024.09

**`Sole contributer`**

<br>

# 프로젝트 요약

**코로나19 환자의 임상 데이터**를 활용하여 환자의 예후(회복 여부)를 예측하는 모델을 개발하고, 여러 머신러닝 알고리즘의 성능을 비교하고자 하였습니다. 이를 위해 sklearn 라이브러리를 통해 **K-최근접 이웃(KNN), 인공 신경망(ANN), 서포트 벡터 머신(SVM), 랜덤 포레스트(Random Forest), XGBoost**와 같은 대표적인 분류 알고리즘을 예측 모델을 구축하고, 모델의 정확도와 성능을 평가하였습니다.

<br>

# 데이터 수집


![image](https://github.com/user-attachments/assets/11cbdc67-06ea-44f9-bc76-e21868950b2b)

- **출처 :**  [COVID-19 Complete Blood Count (CBC) Database (kaggle.com)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-complete-blood-count-clinical-database)
- **데이터 요약** : 방글라데시 다카 의료 대학 병원에서 수집 된 103명의 환자(생존 61명(59.22%), 사망 42명(40.78%))의 임상 매개변수와 병원 입원, 퇴원/사망 결과가 수집
- **12열**: 입원일, 퇴원(사망일), 결과, 나이, 성별, 데이터수집일, 받은 치료 내용, Ventilateor 적용여부, RBC, Monocyte, WBC, PLT, Lymphocyte, Neutrophils
- **103행**: 103명의 환자

<br>

# 데이터 전처리

### 재원일수 계산

- 퇴원일 – 입원일

```python
covid_data['Hospital_Day'] = (covid_data['Discharge_DATE_or_date_of_Death'] - covid_data['Admission_DATE_']).dt.days
```

![image 1](https://github.com/user-attachments/assets/9343a04f-9264-4932-a62c-e8ad9f80fa52)
![image 2](https://github.com/user-attachments/assets/e43d4314-1d84-41b9-aa34-29770d665953)

### 이진 인코딩

- ‘Outcome’열 : Recovered -> 1, Not_recovered -> 0)
- ‘Gender’열 : Male -> 1, Female -> 0
- ‘Ventilated_(Y/N)’열 : Yes -> 1, No -> 0

```python
# # Outcome 열 이진 인코딩 (Recovered -> 1, Not_recovered -> 0)
covid_data['Outcome'] = covid_data['Outcome'].map({'Recovered': 1, 'Not Recovered': 0})

# Gender 열 이진 인코딩 (Male -> 1, Female -> 0)
covid_data['Gender'] = covid_data['Gender'].map({'Male': 1, 'Female': 0})

# Ventilated_(Y/N) 열 이진 인코딩 (Yes -> 1, No -> 0)
covid_data['Ventilated'] = covid_data['Ventilated'].map({'Yes': 1, 'No': 0})
```

![image 3](https://github.com/user-attachments/assets/bf8e07ec-5d72-41a8-a20e-efb2598cbc6a)
![image 4](https://github.com/user-attachments/assets/50333dbb-9baa-44cc-a4c7-1cdca8f34884)

### One-hot 인코딩:

- ‘What kind of Treatment provided’ 열의 값 목록을 다시 열로 만들어 0 또는 1로 인코딩

```python
covid_data['What_kind_of_Treatment_provided_'] = covid_data['What_kind_of_Treatment_provided_'].str.split(',')

mlb = MultiLabelBinarizer()
treatment_encoded = mlb.fit_transform(covid_data['What_kind_of_Treatment_provided_'])

treatment_encoded_df = pd.DataFrame(treatment_encoded, columns=mlb.classes_)

covid_data_encoded = pd.concat([covid_data, treatment_encoded_df], axis=1).drop('What_kind_of_Treatment_provided_', axis=1)
```


![image 5](https://github.com/user-attachments/assets/5d4f6e5f-3a9f-4c9d-a131-4e2a7eea0e25)
![image 6](https://github.com/user-attachments/assets/c92ab270-5c15-495a-9ed0-b88872d2b36d)


<br>

# **탐색적 데이터 분석(EDA)**


## 데이터 분포 및 이상치 확인

![image 7](https://github.com/user-attachments/assets/15c5cffd-5969-4e8e-89c5-1af7e892e67d)

## Pearson 상관관계 매트릭스와 P-value 매트릭스

![image 8](https://github.com/user-attachments/assets/597df392-0b6f-4b0d-be31-a7795bd390fe)

<br>

# 모델 소개



- **인공신경망 (ANN, Artificial Neural Network):** 신경망 모델의 가장 기본적인 형태로, 입력층, 은닉층, 출력층으로 나뉩니다. 각 층은 이전 층의 출력을 입력으로 받아 가중치를 조정하며 학습합니다.
![glossarymultilayered](https://github.com/user-attachments/assets/f12007ab-73f6-45ae-b93c-0af50e31bf32)







- **K-최근접 이웃 (KNN, K-Nearest Neighbors)**: KNN은 가장 가까운 K개의 데이터 포인트를 참고하여 분류하는 모델입니다. 거리에 기반하여 데이터 포인트 간의 유사성을 계산합니다.
![image 9](https://github.com/user-attachments/assets/47d0bf48-fdac-4cd2-8807-a44e7ef12f8c)

- **서포트 벡터 머신 (SVM, Support Vector Machine):** SVM은 데이터 포인트를 두 개의 클래스로 나누는 최적의 경계(결정 경계)를 찾는 알고리즘입니다. 주로 선형 분리 문제에 적합합니다.
![image 10](https://github.com/user-attachments/assets/f14cc67c-d1f1-4296-b2b8-aa5cb31b21f8)

- **랜덤 포레스트 (Random Forest):** 여러 개의 결정 트리(Decision Trees)를 학습시켜 각각의 예측을 결합하여 최종 예측을 수행하는 앙상블 학습 방법입니다.
![image 11](https://github.com/user-attachments/assets/936f0c1f-d169-42ee-91dd-9308fe944dec)
- **XGBoost (Extreme Gradient Boosting):** 결정 트리를 기반으로 앙상블 학습 알고리즘으로, Boosting 기법의 발전된 형태입니다. 하이퍼파라미터 값의 모든 조합을 학습하고, 최적의 조합을 찾습니다. 내장된 폴드 교차 검증 알고리즘으로 최적의 모델을 선택합니다.
![image 12](https://github.com/user-attachments/assets/a3b7e85f-66c5-4fdf-b178-71bf04187af1)

<br>

# Code


## ANN

```python
# 인공 신경망(ANN) 모델 정의 및 학습
ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann_model.fit(X_train, y_train)

# 테스트 데이터셋에 대한 예측 수행
y_pred = ann_model.predict(X_test)
```

## KNN

```python
# KNN 모델 생성 (이웃 5명 기준)
knn_model = KNeighborsClassifier(n_neighbors=5)

# 모델 학습
knn_model.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = knn_model.predict(X_test)
```

## SVM

```python
# 선형 커널을 사용하는 SVM 분류기 초기화 및 재현성을 위한 랜덤 상태 고정
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# 학습 데이터로 모델 학습
svm_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = svm_model.predict(X_test)

```

## Random Forest

```python
# 재현성을 위한 랜덤 상태 고정으로 랜덤 포레스트 분류기 초기화
rf_model = RandomForestClassifier(random_state=42)

# 학습 데이터로 모델 학습
rf_model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = rf_model.predict(X_test)
```

## XGBoost

```python
# 하이퍼파라미터 튜닝을 위한 파라미터 그리드 정의
param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],  # 학습률 값
    'max_depth': [3, 5, 7],             # 트리의 최대 깊이
    'n_estimators': [50, 100, 200]      # 부스팅 라운드 수
}

# XGBoost 분류기 초기화
xgb_model = xgb.XGBClassifier()

# 교차 검증과 함께 그리드 서치를 설정
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# 그리드 서치를 사용하여 학습 데이터에 모델 학습
grid_search.fit(X_train, y_train)

```

# 학습 과정

---

|  | ANN | KNN | SVM | Random Forest | XGBoost |
| --- | --- | --- | --- | --- | --- |
| Hyperparameter | •hidden_layer_sizes=(100,) •max_iter=1000 •random_state=42 |•n_neighbors=5 |  •kernel='linear’  | •random_state=42 | •param_grid = {'learning_rate': [0.05, 0.1, 0.2,'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200] }•cv = 3 •random_state=42 | 





# 성능 평가 방식



- 전체 데이터셋을 학습용 :평가용=8: 2로 분할합니다.
- 학습 종료 후 Test data를 예측하도록 한 뒤, 정답 여부 확인합니다.
- Confusion matrix 생성 후 Precision, Recall, F1-score, AUC를 계산하였습니다.

![image 14](https://github.com/user-attachments/assets/95091072-e148-4e4a-97a3-e41a6d846ad6)
![image 13](https://github.com/user-attachments/assets/efb9c8d2-ef7c-4bc6-88b3-e636244ba940)
![image 15](https://github.com/user-attachments/assets/2d08456f-bea3-41b2-8110-a4f48b809abd)
![image 16](https://github.com/user-attachments/assets/d26540eb-f110-4083-9655-da60fe382307)

<br>

# 결과


## ANN

|  | TEST 결과 |
| --- | --- |
| Recovered | •Precision: 0.75  •Recall: 0.23 •F1-score: 0.53 |
| Not Recovered | •Precision: 0.52 •Recall: 0.92 •F1-score: 0.67 |

![image 17](https://github.com/user-attachments/assets/a26ddd54-c598-493b-95d5-b6b18796e11b)

## KNN

|  | TEST 결과 |
| --- | --- |
| Recovered | •Precision: 0.75  •Recall: 0.23 •F1-score: 0.53 |
| Not Recovered | •Precision: 0.52 •Recall: 0.92 •F1-score: 0.67 |

![image 18](https://github.com/user-attachments/assets/5a81ddad-2d1b-4c5a-9018-d0798e82490a)

## SVM

|  | TEST 결과 |
| --- | --- |
| Recovered | •Precision: 0. 92  •Recall: 0.85 •F1-score: 0.88 |
| Not Recovered | •Precision: 0.88  •Recall: 0.88  •F1-score: 0.88 |

![image 19](https://github.com/user-attachments/assets/f4e098de-90f1-4c51-b5cb-42bf5e8edfdc)

## Random Forest

|  | TEST 결과 |
| --- | --- |
| Recovered | •Precision: 1.00 •Recall: 1.00 •F1-score: 1.00 |
| Not Recovered | •Precision: 1.00  •Recall: 1.00 •F1-score: 1.00 |

![image 20](https://github.com/user-attachments/assets/40938f6b-ff73-43da-8b3f-1ea501652957)

## XGBoost

|  | TEST 결과 |
| --- | --- |
| Recovered | •Precision: 1.00 •Recall: 0.85 •F1-score: 0.92 |
| Not Recovered | •Precision: 0.86 •Recall: 1.00 •F1-score: 0.92 |

![image 21](https://github.com/user-attachments/assets/565ada4d-c332-41ad-b707-a72b7fadbaa6)















