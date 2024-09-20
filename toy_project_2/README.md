# 암 발생률의 지역간 격차와 요인 분석

2024.09 - 2024.09

**`Sole contributer`**



# 프로젝트 요약



![image](https://github.com/user-attachments/assets/b495e218-ebb0-4ec8-a47a-8a9fc99b49cd)
**`데이터 원본: KOSIS '22대 분류별 진료현황'`**

![2671a043-5bf7-49cd-91eb-6e5642dda072](https://github.com/user-attachments/assets/5bceda64-e739-4583-ba6b-5160f002989e)
![**`데이터 원본: KOSIS '사망원인(237항목)/성/연령별 사망자수, 사망률'`**](image.png)

**`데이터 원본: KOSIS '사망원인(237항목)/성/연령별 사망자수, 사망률'`**

암은 국내 주요 사망 원인 중 하나로, 다양한 요인에 의해 발생할 수 있습니다. 이 프로젝트는 KOSIS 암 등록 통계([1. 시군구 시기/24개 암종/성별 암발생자수, 상대빈도, 조발생률, 연령표준화발생률 (kosis.kr)](https://kosis.kr/statHtml/statHtml.do?orgId=117&tblId=DT_117N_A11109&vw_cd=MT_ZTITLE&list_id=F_35&scrId=&seqNo=&lang_mode=ko&obj_var_id=&itm_id=&conn_path=MT_ZTITLE&path=%252FstatisticsList%252FstatisticsListIndex.do))를 분석 및 시각화하여, **지역간 발생를의 격차가 큰 암종을 확인**하고 요인을 분석하기 위해 진행되었습니다 .


<br>

# 데이터 수집



## **KOSIS 암 등록 통계**

- **출처 :** [1. 시군구 시기/24개 암종/성별 암발생자수, 상대빈도, 조발생률, 연령표준화발생률 (kosis.kr)](https://kosis.kr/statHtml/statHtml.do?orgId=117&tblId=DT_117N_A11109&vw_cd=MT_ZTITLE&list_id=F_35&scrId=&seqNo=&lang_mode=ko&obj_var_id=&itm_id=&conn_path=MT_ZTITLE&path=%252FstatisticsList%252FstatisticsListIndex.do))
- **시기** : 1999-2003년,  2004-2008년,  2009-2013년,  2014-2018년
- **시/도** : 서울특별시 등 17개 행정구역(세종특별자치시는 1999-2003년,  2004-2008년 데이터 없음)
- **27열:** 시기, 지역, 모든 암, 간암, 갑상선암 등 24개 암종
- **72행**: 4개의 조사구간 별로 17개 행정구역
- **연령표준화 발생률(값)** :우리나라 2020년 주민등록 인구를 표준인구로 사용하여 산출(명/10만명)

## **질병관리청 지역사회건강조사**

- **출처:** [시도별 주요결과 < 질병관리청 지역사회건강조사 (kdca.go.kr)](https://chs.kdca.go.kr/chs/recsRoom/ctprvnResultMain.do))
- **시기** : 2009-2013년,  2014-2018년
- **시/도** : 서울특별시 등 17개 행정구역(세종특별자치시는 2009-2013년 데이터 없음)
- **10열** : 시기, 지역, 삶의 질, 당뇨, 고혈압, 스트레스 인지율, 걷기실천율, 월간 음주율, 현재 흡연률, 저염식 선호율)
- **72행** : 2개의 조사구간 별로 17개 행정구역
- **표준화율(값):** 인구구성 차이에 따른 영향을 표준인구로 보정한 결과

<br>

# 데이터 전처리

## **KOSIS 암 등록 통계**

### 결측치 처리

> 열의 평균으로 처리
> 

```python
# 숫자형 열의 평균으로 결측치 채우기
numeric_columns = cancer_data.select_dtypes(include=['float64', 'int64'])
cancer_data[numeric_columns.columns] =numeric_columns.fillna(numeric_columns.mean())
```

![image 1](https://github.com/user-attachments/assets/99b9a50b-66e1-44c3-9a47-241c4fa07cdf)

### **데이터 스케일링**

> Robust Scaler
> 

```python
scaler = RobustScaler()
```

## **질병관리청 지역사회건강조사**

### 데이터 병합

> 지역별 암 발생률 데이터와 inner join
> 

```python
# 시기와 지역이 같은 데이터 조인하기
merged_data = pd.merge(cancer_data, factor_data, on=['시기', '지역'], how='outer')
```

![image 2](https://github.com/user-attachments/assets/5c93a9e4-5692-4314-9889-df1917f8d85e)
### 결측치 처리

> 시각화 후 1) 열의 평균으로 처리 또는 2)행 삭제
> 

```python
# null이 2개이상인 행 삭제
merged_data = merged_data[merged_data.isnull().sum(axis=1) < 2]

# 2009-2013 제주도의 값이 없는 호지킨림프종은 mean으로
merged_data['호지킨 림프종(C81)'] = merged_data['호지킨 림프종(C81)'].fillna(merged_data['호지킨 림프종(C81)'].mean())
```

![image 3](https://github.com/user-attachments/assets/6a5e0e87-d1a8-45de-abf7-23f8e7c89e5f)

<br>

# 데이터 분석


## **KOSIS 암 등록 통계**

- 변동계수 계산 (Coefficient of Variation, CV) 하여 지역 간 편차 확인

```python
range= cancer_data[cancer_columns].std() / cancer_data[cancer_columns].mean()
```

- Tableau 맵 차트를 통해 암종/시기에 따른 연령표준화발생률 시각화
- 시기/ 암종/지역별로 연령표준화발생률의 분포와 변화를 확인할 수 있도록 대시보드 생성

![e1506d11-b85b-4aad-a9e3-9b812ef56b2e](https://github.com/user-attachments/assets/5eb69976-f2ea-44b0-8bb2-6204d47c5fa1)


 **`tableau cloud에서 제공하는 javascript API를 활용해 웹페이지 임베딩`**

## **질병관리청 지역사회건강조사**

- Kendall Tau 상관계수 및 p-value Matrix 시각화

```python
for col1 in columns:
    for col2 in columns:
        corr, p_value = kendalltau(corr_data[col1], corr_data[col2])
        kendall_corr_matrix.loc[col1, col2] = corr
        p_value_matrix.loc[col1, col2] = p_value

```
<br>

# 결과


## **암 발생지도(변동계수 1-3위)**

![image 4](https://github.com/user-attachments/assets/473d8ede-2390-4e68-be4f-b028f9211f1b)
<div style="display: flex; flex-direction: row;">
    <h4 style="margin-right: 80px;">  1. 갑상선암 맵차트 </h4> 
    <h4 style="margin-right: 80px;"> 2. 폐암 맵차트  </h4> 
    <h4 style="margin-right: 80px;">    3. 자궁체부암 맵차트      </h4> 
</div>

## 변동계수 순위

![image 5](https://github.com/user-attachments/assets/be345d7f-e8a0-4b58-891e-eba35720778c)

## 

## Kendall Tau 상관계수 및 p-value Matrix 시각화

![image 6](https://github.com/user-attachments/assets/28b3e43f-7f3e-4ad9-acab-7f7a53a16dc6)

- 시/도별로 발생률의 편차가 가장 심한 암종은 ‘**갑상선암(C73)’(강원 115.3~ 전남 268.7)**’으로 확인되었고, 그 다음으로 ‘**폐암(C33~C34)(‘제주 210.3~ 경북 276.1)**과 **‘자궁체부암(C54)’(경남 13.10~서울 19.5)**으로 나타났습니다.
- ‘**자궁체부암(C54)‘**과 ‘**당뇨**’가 Kendall Tau 상관계수 +0.53, P-value <0.001로 **유의미한 중간 정도의 상관관계**를 보였습니다.
- **갑상선 암의 큰 지역 편차**를 설명하는 요인은 확인할 수 없었으나 , 기사를 통해 “국내 시도별 갑상선암 발생률은, 시도별 **갑상선암 검진율과 강한 상관관계를** 보였다＂는 복지부의 분석 결과를 찾을 수 있었습니다. (출처 :[암발생률, 지역별로 최대 15배 '격차' < 보건복지 < 정책 < 기사본문 - 메디칼업저버](http://www.monews.co.kr/news/articleView.html?idxno=95139) [(monews.co.kr)](http://www.monews.co.kr/news/articleView.html?idxno=95139))








