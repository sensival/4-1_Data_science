#### 점수에 따른 pass/fail 여부 로지스틱 회귀

import pandas as pd
import statsmodels.api as sm

score = [56, 60, 61, 67, 69, 55, 70, 44, 51, 64, 60, 50, 
         68, 72, 90, 93, 85, 74, 81, 88, 92, 97, 77, 78, 98]
_pass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]

result = pd.DataFrame(
    {"score": score, "_pass": _pass}
)



logis = sm.Logit.from_formula('_pass ~ score', result).fit()
print(logis.summary())

import matplotlib.pyplot as plt
import numpy as np

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 점수(score) 범위 생성
x_values = np.linspace(min(result['score']), max(result['score']), 100)

# 로지스틱 회귀 모델의 예측 결과 계산
y_values = sigmoid(logis.params[0] + logis.params[1] * x_values)

# 데이터 시각화
plt.scatter(result['score'], result['_pass'], color='blue', label='Data')
plt.plot(x_values, y_values, color='red', label='Logistic Regression')
plt.xlabel('Score')
plt.ylabel('Pass/Fail')
plt.title('Logistic Regression: Score vs Pass/Fail')
plt.legend()
plt.show()