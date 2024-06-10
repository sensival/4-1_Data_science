import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

# 기본 리픗롯
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips)
plt.show()




# autofmt_xdate() x축 레이블이 서로 겹치지 않고 잘 읽을 수 있도록 회전시키거나 간격을 조정, 주로 x축에 날짜 데이터가 포함된 경우 유용하게 사용
df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)

fig = g.fig
ax = g.ax

fig.autofmt_xdate() 
plt.show()




# 다차원 그래프 
sns.catplot(x="day", y="total_bill", hue="smoker",
            col="time", aspect=.6,
            kind="swarm", data=tips)
plt.show()




# 박스플롯
titanic = sns.load_dataset("titanic")
g = sns.catplot(x="fare", y="survived", row="class",
                kind="box", orient="h", height=1.5, aspect=4,
                data=titanic.query("fare > 0"))
g.set(xscale="log");
plt.show()



# 2개 요소씩 짝지어서 그래표 표현, 대각선은 분포히스토그램
iris = sns.load_dataset("iris")

df = sns.load_dataset("iris")
df.head(100)

sns.pairplot(iris)
plt.show()


# pair Grid
g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
plt.show()


# 히트맵
# seaborn 스타일 설정
sns.set(style="whitegrid")

# flights 데이터셋 로드
flights = sns.load_dataset("flights")

# 'month' 열을 카테고리형으로 변환하고 순서 지정
months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
flights['month'] = pd.Categorical(flights['month'], categories=months_order, ordered=True)

# flights 데이터프레임을 피봇 테이블로 변환
# 행: month, 열: year, 값: passengers
flights_pivot = flights.pivot(index="month", columns="year", values="passengers")

# 피봇된 데이터 확인
print(flights_pivot)

# 그래프 크기 설정
plt.figure(figsize=(10, 10))

# 히트맵 생성, 각 셀에 데이터를 텍스트로 표시하고, 텍스트 형식을 정수로 지정
ax = sns.heatmap(flights_pivot, annot=True, fmt="d", cmap="YlGnBu")

# 그래프 표시
plt.show()

# 회귀데이터셋
sns.set(style="ticks")

df = sns.load_dataset("anscombe")
df.head(10)

sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

plt.show();
# sns.lmplot: 선형 회귀 플롯을 생성합니다. x와 y 변수로 플롯을 그리고, col 및 hue를 통해 각 데이터셋을 다른 색상과 열로 나누어 표시합니다.
# x="x", y="y": x와 y 데이터를 플롯의 x축과 y축에 매핑합니다.
# col="dataset", hue="dataset": dataset 컬럼을 기준으로 여러 플롯을 생성하고, 색상을 다르게 지정합니다.
# data=df: 데이터셋으로 df를 사용합니다.
# col_wrap=2: 2개의 열로 플롯을 감싸서 배치합니다.
# ci=None: 회귀선 주위의 신뢰 구간을 표시하지 않습니다.
# palette="muted": muted 팔레트를 사용하여 색상을 설정합니다.
# height=4: 플롯의 높이를 설정합니다.
# scatter_kws={"s": 50, "alpha": 1}: 점의 크기(s=50)와 투명도(alpha=1)를 설정합니다.
# plt.show(): 플롯을 화면에 표시합니다.




# 산점도 색조, 크기 변경
sns.set(style="white")

mpg = sns.load_dataset("mpg")

sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=mpg)

plt.show()




# 선그래프에 신뢰구간 추가
sns.set(style="darkgrid")

fmri = sns.load_dataset("fmri")

df = sns.load_dataset("fmri")
print(df.head(10))

sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri, ci=None)
plt.show();








