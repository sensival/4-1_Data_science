# 유의어 사이의 관계를 그래프로 정의하고 있는 방대한 데이터셋
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
 
print(wordnet.synsets('man'))

man = wordnet.synset('man.n.01')
print(man.definition()) 
# an adult person who is male (as opposed to a woman)

man2 = wordnet.synset('man.n.02')
print(man2.definition()) 
# someone who serves in the armed forces; a member of a military force


boy = wordnet.synset('boy.n.01')
guy = wordnet.synset('guy.n.01')
girl = wordnet.synset('girl.n.01')
woman = wordnet.synset('woman.n.01')
 
print(man.path_similarity(man))    
print(man.path_similarity(boy))     
print(man.path_similarity(guy))    
print(man.path_similarity(girl))    
print(man.path_similarity(woman))   


# konlpy
from konlpy.tag import Kkma
from konlpy.utils import pprint
 
kkma = Kkma()
 
sentence = u'영희와 철수는 백구를 산책시키기 위해 한강에 갔다. 한강에 도착하여 누렁이를 만났다.'
 
print('형태소 : ' + str(kkma.morphs(sentence)))
 
print('명사 : ' + str(kkma.nouns(sentence)))
 
print('품사 : ' + str(kkma.pos(sentence)))

