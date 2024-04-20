

###### data 입력 받기 무한히 입력
import sys 
for line in sys.stdin: 
    line = line.strip() 
    words = line.split() 
    for word in words: 
        print ('%s\t%s' % (word, 1))

###### 맵리듀스
import sys
current_word = None
current_count = 0
word = None
for line in sys.stdin:
    line = line.strip()
    # parse the input we got from mapper.py
    word, count = line.split('\t', 1) # 한번만 분할해서 word 와 count에 각각넣어라
    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue
    if current_word == word:
        current_count += count
    else:
        if current_word:
            print ('%s\t%s' % (current_word, current_count))
        current_count = count
        current_word = word
if current_word == word:
    print ('%s\t%s' % (current_word, current_count))
