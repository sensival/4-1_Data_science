import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import kstest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc

# 데이터셋 로드
covid_data = pd.read_csv("C:/Users/wogns/OneDrive/바탕 화면/깃허브 레포지토리/Data_study/데이터사이언스/toy_project_2/encoded_data.csv")


plt.rcParams['font.family'] = 'Malgun Gothic'  


def train_knn(X_train, X_test, y_train, y_test):
    # KNN 모델 생성 (이웃 5명 기준)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    # 모델 학습
    knn_model.fit(X_train, y_train)
    # 테스트 데이터로 예측 수행
    y_pred = knn_model.predict(X_test)
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN 정확도:", accuracy)
    print("KNN 분류 보고서:")
    print(classification_report(y_test, y_pred))
    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # 혼동 행렬 시각화
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("KNN Confusion Matrix(Recovered -> 1, Not_recovered -> 0)")
    plt.xlabel("예측된 라벨")
    plt.ylabel("실제 라벨")
    plt.show()

    y_pred_proba = knn_model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률값
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # FPR, TPR 계산
    roc_auc = auc(fpr, tpr)  # AUC 계산

    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


def train_ann(X_train, X_test, y_train, y_test):
    # 인공 신경망(ANN) 모델 정의 및 학습
    ann_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    ann_model.fit(X_train, y_train)

    # 테스트 데이터셋에 대한 예측 수행
    y_pred = ann_model.predict(X_test)

    # 정확도 계산 및 출력
    accuracy = accuracy_score(y_test, y_pred)
    print("ANN 정확도:", accuracy)

    # 분류 보고서 출력
    print("ANN 분류 보고서:")
    print(classification_report(y_test, y_pred))

    # 혼동 행렬 생성 및 시각화
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("ANN Confusion Matrix(Recovered -> 1, Not_recovered -> 0)")
    plt.xlabel("예측된 라벨")
    plt.ylabel("실제 라벨")
    plt.show()

    y_pred_proba = ann_model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률값
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # FPR, TPR 계산
    roc_auc = auc(fpr, tpr)  # AUC 계산

    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ANN ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_svm(X_train, X_test, y_train, y_test):
    # 선형 커널을 사용하는 SVM 분류기 초기화 및 재현성을 위한 랜덤 상태 고정
    svm_model = SVC(kernel='linear', probability=True, random_state=42)

    # 학습 데이터로 모델 학습
    svm_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측 수행
    y_pred = svm_model.predict(X_test)

    # 테스트 데이터에 대한 모델의 정확도 계산 및 출력
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM 정확도:", accuracy)

    # 상세한 분류 보고서 출력
    print("SVM 분류 보고서:")
    print(classification_report(y_test, y_pred))

    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 히트맵을 사용하여 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("SVM Confusion Matrix(Recovered -> 1, Not_recovered -> 0)")
    plt.xlabel("예측된 라벨")
    plt.ylabel("실제 라벨")
    plt.show()

    # 확률 예측 수행
    y_pred_proba = svm_model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률값
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # FPR, TPR 계산
    roc_auc = auc(fpr, tpr)  # AUC 계산

    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_random_forest(X_train, X_test, y_train, y_test):
    # 재현성을 위한 랜덤 상태 고정으로 랜덤 포레스트 분류기 초기화
    rf_model = RandomForestClassifier(random_state=42)

    # 학습 데이터로 모델 학습
    rf_model.fit(X_train, y_train)

    # 테스트 데이터에 대한 예측 수행
    y_pred = rf_model.predict(X_test)

    # 테스트 데이터에 대한 모델의 정확도 계산 및 출력
    accuracy = accuracy_score(y_test, y_pred)
    print("랜덤 포레스트 정확도:", accuracy)

    # 상세한 분류 보고서 출력
    print("랜덤 포레스트 분류 보고서:")
    print(classification_report(y_test, y_pred))

    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 히트맵을 사용하여 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Random forest Confusion Matrix(Recovered -> 1, Not_recovered -> 0)")
    plt.xlabel("예측된 라벨")
    plt.ylabel("실제 라벨")
    plt.show()

    
    # ROC 곡선 및 AUC 계산
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률값
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # FPR, TPR 계산
    roc_auc = auc(fpr, tpr)  # AUC 계산

    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random forest ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


    

def train_xgboost_with_gridsearch(X_train, X_test, y_train, y_test):
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

    # 그리드 서치에서의 최적 파라미터와 최적 점수 출력
    print("최적 파라미터:", grid_search.best_params_)
    print("최적 점수:", grid_search.best_score_)

    # 그리드 서치에서 최적 모델을 가져옴
    best_model = grid_search.best_estimator_

    # 테스트 데이터에 대한 라벨 예측
    y_pred = best_model.predict(X_test)

    # 모델의 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost 정확도:", accuracy)

    # 분류 보고서 출력
    print("XGBoost 분류 보고서:")
    print(classification_report(y_test, y_pred))

    # 혼동 행렬 계산
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("XGBoost Confusion Matrix(Recovered -> 1, Not recovered -> 0)")
    plt.xlabel("예측된 라벨")
    plt.ylabel("실제 라벨")
    plt.show()

    # ROC 곡선 및 AUC 계산
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 양성 클래스에 대한 확률값
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # FPR, TPR 계산
    roc_auc = auc(fpr, tpr)  # AUC 계산

    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 대각선 선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

########################### 모델 학습 #############################
target_df = covid_data['Outcome']

# 입력 특성을 얻기 위해 'Biopsy' 열 삭제
input_df = covid_data.drop(columns=['Outcome'])

# 입력 데이터를 NumPy 배열로 변환하고 'float32' 데이터 타입으로 캐스팅
X = np.array(input_df).astype('float32')

# 타겟 데이터를 NumPy 배열로 변환하고 'float32' 데이터 타입으로 캐스팅
y = np.array(target_df).astype('float32')

# SMOTE(합성 소수 클래스 과샘플링 기법)를 적용하여 데이터셋 균형 맞추기
# SMOTE는 소수 클래스의 샘플 수를 증가시켜 균형 분포를 달성합니다.
smote = SMOTE(random_state=42)  # 재현성을 위한 고정된 랜덤 시드로 SMOTE 초기화
X_resampled, y_resampled = smote.fit_resample(X, y)  # SMOTE를 사용하여 균형 잡힌 데이터셋 생성

# 데이터셋을 학습 및 테스트 세트로 분할
# 학습에 80%, 테스트에 20% 할당
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
# 'random_state=42'는 무작위 분할의 재현성을 보장

# 모델 학습 및 평가
train_ann(X_train, X_test, y_train, y_test)
train_knn(X_train, X_test, y_train, y_test)
train_svm(X_train, X_test, y_train, y_test)
train_random_forest(X_train, X_test, y_train, y_test)
train_xgboost_with_gridsearch(X_train, X_test, y_train, y_test)






