# 정수 인코딩
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text="A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
text=sent_tokenize(text)
print(text)

# 정제와 단어 토큰화
vocab={} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence=word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word=word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    sentences.append(result) 
print(sentences)


# 빈도수 높은 수대로 정렬
vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)


# 단어에 인덱스를 부여
# 기준 : 높은 빈도수 단어에 작은 수의 인덱스 부여
word_to_index={}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word]=i
print(word_to_index)


vocab_size=5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거-->  word_to_index 딕셔너리에는 5개만 남음
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제


word_to_index['OOV']=len(word_to_index)+1
#   Out-Of-Vocabulary(단어 집합에 없는 단어, 딕셔너리에 있는 5개 제외) 'OOV'란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 'OOV'의 인덱스(이경우엔 6)로 인코딩


encoded=[]
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)



