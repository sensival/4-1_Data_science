
import pandas as pd
import numpy as np
data = pd.read_csv('application_train.csv')

# 컬럼별 type 확인 및 결측치 확인
print('data information')
data.info()
print('data information of NAN')
data.isnull().sum()


# 문자전환
data = data.replace(' ', '')

# 만약 결측치가 문자열 스페이스(' ')로 되어 있다면, np.nan으로 바꾸어 Pandas 라이브러리가 인식할수 있도록 변환
data = data.replace('', np.nan)

# 결측 row 제거하는 방법
data.dropna(how='all') # 'all':한 행이 모두 missing value이면 제거, any': 행 내에서 하나라도

# 결측치 처리
data.fillna(0, inplace=True)

# 처리 방식에 따른 컬럼 정리
category_list = ['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY']
one_hot_list = ['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE',
                'FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','CODE_GENDER','EMERGENCYSTATE_MODE']


# categorize
categories_encoded = pd.DataFrame()
cate_cols = []

print('카테고리화')
for x in category_list:
    print(x)
    X = data[x]
    x_encoded, x_categories = X.factorize()

    # dataframe
    temp_df = pd.DataFrame(x_encoded)
    categories_encoded = pd.concat([categories_encoded,temp_df],axis=1)

    # 컬럼명 추가
    cate_cols.append(x + '_1')

# 컬럼명 수정
categories_encoded.columns = [cate_cols]

print("카테고리화 된 리스트 head")
categories_encoded.head()



# 항목별 맥스값 체크 (확인용)
print("카테고리화 된 리스트 확인용 MAX값 출력")
for x in category_list:
    print(x)
    col = x + '_1'
    print(col)
    print(max(categories_encoded[col].values))
    
    print(x, ' max : ', max(categories_encoded[col].values)[0])



from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

# One-Hot-Encoder
onehot_encoded = pd.DataFrame()
onehot_cols = []
print('one-hot encoding')

for x in one_hot_list:    
    print(x)
    X = data[x]
    x_encoded, x_categories = X.factorize()
    x_1hot = encoder.fit_transform(x_encoded.reshape(-1,1)) # reshape(-1,1)을 사용하여 1차원 배열을 2차원 배열로 변환하고, 그 다음 fit_transform() 메서드를 사용하여 원-핫 인코딩을 수행
    x_1hot = x_1hot.toarray()    
    print(x_1hot)

    # dataframe
    temp_df = pd.DataFrame(x_1hot)
    onehot_encoded = pd.concat([onehot_encoded,temp_df],axis=1)

    # 컬럼명 추가
    for i in range(1, temp_df.shape[1] +1):
        onehot_cols.append(x + '_' + str(i))

# 컬럼명 수정
onehot_encoded.columns = [onehot_cols]

# 항목별 맥스값 체크 (확인용)
for x in one_hot_list:
    col = x + '_1'
    print(x, ' max : ', max(onehot_encoded[col].values)[0])

# 모두 1이면 정상

