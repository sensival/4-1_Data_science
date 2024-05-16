###### 키와 몸무게 선형 회귀   

import pandas as pd
import statsmodels.api as sm

height = [170, 168, 177, 181 ,172, 171, 169, 175, 174, 178, 170, 167,
          177, 182 ,173, 171, 170, 179, 175, 177, 186, 166, 183, 168]
weight = [70, 66, 73, 77, 74, 73, 69, 79, 77, 80, 74, 68, 71, 76, 78, 
          72, 68, 79, 77, 81, 84, 73, 78, 69]

body = pd.DataFrame(
    {'height': height,
    'weight': weight
    }
)


reg = sm.OLS.from_formula("height ~ weight", body).fit()
# height ~ weight: height를 종속 변수로, weight를 독립 변수로 설정
# body: 데이터프레임으로, height와 weight 열을 포함
print(reg.summary())

# 시각화
import matplotlib.pyplot as plt

# 산점도 그리기
plt.scatter(body['weight'], body['height'], label='Data')

# 회귀선 그리기
plt.plot(body['weight'], reg.predict(body['weight']), color='red', label='Regression Line')

# 축과 레이블 설정
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Height vs Weight Regression')
plt.legend()

# 그래프 보여주기
plt.show()


#### 당뇨병 데이터로 선형 회귀

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

## 사이킷런 linear regression model 사용
model_diabetes = LinearRegression().fit(X, y)
predictions = model_diabetes.predict(X)

plt.scatter(y, predictions)
plt.xlabel("Actual Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression")
plt.title("Correlation between Actual and Predicted Diabetes Progression")
plt.show()

## statsmodels OLS regression model 사용
dfX0 = pd.DataFrame(X, columns=diabetes.feature_names)
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(y, columns=["Progression"])

model_diabetes_sm = sm.OLS(dfy, dfX)
result_diabetes_sm = model_diabetes_sm.fit()
print(result_diabetes_sm.summary())

# 잔차의 합
result_diabetes_sm.resid.plot()
plt.title("Residual Vector")
plt.xlabel("Data Number")
plt.show()
print("Sum of residuals:", result_diabetes_sm.resid.sum()) 


## 모델로 결과 예측
x_new = np.mean(X, axis=0)

# dfX0을 [:, np.newaxis] 를 통해 2차원 배열로 바꾼 후 .T로 하나의 행을 가진 데이터로 바꿈
dfx_new = sm.add_constant(pd.DataFrame(np.array(x_new)[:, np.newaxis].T, columns=diabetes.feature_names), has_constant="add") # has_constant="add" 상수항 열을 추가


predictions_new = result_diabetes_sm.predict(dfx_new)
y_mean = np.mean(y)


print("Model Parameters:")
print(result_diabetes_sm.params)

# Plot the residual vector
result_diabetes_sm.resid.plot()
plt.title("Residual Vector")
plt.xlabel("Data Number")
plt.show()


""" 보스턴 주택 가격 데이터 사라짐
#### 싸이킷런 보스턴 주택 가격 데이터로 선형 회귀 분석
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 


boston = load_boston()
X = boston.data
y = boston.target
model_boston = LinearRegression().fit(boston.data, boston.target)

predictions = model_boston.predict(boston.data)

plt.scatter(boston.target, predictions)
plt.xlabel("price")
plt.ylabel("prediction")
plt.title("co-relation")
plt.show()


model = sm.OLS(y, X) # Ordinary Least Squares(OLS) 회귀 분석
result = model.fit()
print(result.summary())


dfX0 = pd.DataFrame(boston.data, columns=boston.feature_names)
dfX = sm.add_constant(dfX0)
dfy = pd.DataFrame(boston.target, columns=["MEDV"])

model_boston2 = sm.OLS(dfy, dfX)
result_boston2 = model_boston2.fit()
print(result_boston2.summary())




import numpy as np 
x_new = dfX0.mean().values # 1차원 배열
dfx_new = sm.add_constant(pd.DataFrame(np.array(x_new)[:, np.newaxis].T,columns=boston.feature_names),has_constant="add")
# dfX0을 [:, np.newaxis] 를 통해 2차원 배열로 바꾼 후 .T로 하나의 행을 가진 데이터로 바꿈


result_boston2.predict(dfx_new)
dfy.mean()
print(result_boston2.params)
result_boston2.resid.plot()
plt.title("resid vector")
plt.xlabel("data num")
plt.show()
print(result_boston2.resid.sum())  #잔차의 합은 0이됨



"""