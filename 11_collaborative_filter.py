import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings; warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from math import sqrt 

link_small = pd.read_csv('links_small.csv')
link_small = link_small[link_small['tmdbId'].notnull()]['tmdbId'].astype('int') 

md = pd. read_csv('movies_metadata.csv')

md = md.drop([19730, 29503, 35587]) 
md['id'] = md['id'].astype('int') 
smd = md[md['id'].isin(link_small)]
print(smd.shape)

# 데이터 준비
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

# tfidf 벡터라이저 초기화 시 올바른 min_df 값 설정
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
# 또는 min_df=0.01와 같이 설정할 수 있음

# tfidf 매트릭스를 생성
tfidf_matrix = tf.fit_transform(smd['description'])
print(tfidf_matrix.shape)


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]


smd = smd.reset_index()
titles = smd['title']
indces = pd.Series(smd.index, index=titles)


#smd에 인덱스를 포함하고 타이틀을 만든다. pd.Series를 통해서 타이틀을 인덱스로 하고 indces를 만듭니다. 

def getrecommandations(title):
    index = indces[title]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores] 
    return titles.iloc[movie_indices]


list(enumerate(cosine_sim[0]))
getrecommandations('The Dark Knight').head(10)


critics={
     'hhd':{'guardians of the galaxy 2':5,'christmas in august':4,'boss baby':1.5},
     'chs':{'christmas in august':5,'boss baby':2},
     'kmh':{'guardians of the galaxy 2':2.5,'christmas in august':2,'boss baby':1},
     'leb':{'guardians of the galaxy 2':3.5,'christmas in august':4,'boss baby':5}
}

def sim(i,j):
    return sqrt(pow(i,2)+pow(j,2))

for i in critics:
    if i!='chs':
        num1 = critics.get('chs').get('christmas in august')- critics.get(i).get('christmas in august')
        num2 = critics.get('chs').get('boss baby')- critics.get(i).get('boss baby')
        print(i," : ", 1/(1+sim(num1,num2))) #정규화


def sim_distance(data, name1, name2):
    sum=0
    for i in data[name1]:
        if i in data[name2]: #같은 영화를 봤다면
            sum+=pow(data[name1][i]- data[name2][i],2)
        
    return 1/(1+sqrt(sum))


def top_match(data, name, index=3, sim_function=sim_distance):
    li=[]
    for i in data:
        if name!=i: #자기 자신은 제외한다
            li.append((sim_function(data,name,i),i)) # 유사도, 이름을 튜플에 묶어 리스트에 추가한다
    li.sort() #오름차순 정렬
    li.reverse() #내림차순 정렬
    
    return li[:index]

top_match(critics, 'chs')
# 하나의 함수로 여러개의 아이템을 한 번에 비교했고, 
# chs과 hhd의 거리가 가장 가까운 것을 확인할 수 있다. 

