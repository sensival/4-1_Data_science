#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8

from selenium import webdriver 
from bs4 import BeautifulSoup 
import requests 
from selenium.webdriver.common.desired_capabilities import  DesiredCapabilities 
import time 
from selenium.webdriver.common.keys import Keys 
import datetime as dt 
import pandas as pd
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


#웹드라이버 설정
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))


# 시작 날짜 및 종료 날짜 설정
startdate = dt.date(year=2019, month=7, day=1)  # 시작 날짜
untildate = dt.date(year=2019, month=7, day=2)  # 시작 날짜 + 1
enddate = dt.date(year=2019, month=7, day=2)  # 종료 날짜

# 검색어 설정
query = "loopy"
totaltweets = []  # 전체 트윗 저장 리스트
totalfreq = []  # 날짜별 빈도수 저장 리스트

# 종료 날짜와 시작 날짜가 같지 않을 때까지 반복
while not enddate == startdate:
    # 트위터 검색 페이지 URL 생성
    url = 'https://twitter.com/search?q=' + query + '%20since%3A' + str(startdate) + '%20until%3A' + str(untildate) + '&amp;amp;amp;amp;amp;amp;lang=eg'
    driver.get(url)  # URL 열기
    html = driver.page_source  # 페이지 소스 가져오기
    soup = BeautifulSoup(html, 'html.parser')  # BeautifulSoup을 이용해 HTML 파싱

    lastHeight = driver.execute_script("return document.body.scrollHeight")  # 스크롤 높이 저장
    while True:
        dailyfreq = {'date': startdate}  # 해당 날짜

        wordfreq = 0  # 단어 빈도수 초기화
        tweets = soup.find_all("p", {"class": "TweetTextSize"})  # 트윗들을 찾아옴
        # <p>태그를 가지고 class가 tweetTextSize(트위터에서 트윗들의 본문을 담당하는 요소들은 일반적으로 "TweetTextSize"라는 클래스를 가지고 있습니다)인 내용을 가져옴 (CSS)
        

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 페이지 스크롤 다운
        time.sleep(1)  # 1초 대기 페이지 로딩 기다리기

        newHeight = driver.execute_script("return document.body.scrollHeight")  # 새로운 스크롤 높이

        # 스크롤이 내려가고있으면 계속 업데이트
        if newHeight != lastHeight:
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            tweets = soup.find_all("p", {"class": "TweetTextSize"})
            wordfreq = len(tweets)  # 단어 빈도수 계산
        else:
            totalfreq.append(dailyfreq)  # 날짜 추가
            wordfreq = 0  # 단어 빈도수 초기화
            startdate = untildate  # 시작 날짜 변경
            untildate += dt.timedelta(days=1)  # 다음 날짜로 변경
            dailyfreq = {}  # dailyfreq 초기화
            totaltweets.append(tweets)  # 트윗 리스트에 트윗 추가
            break

        lastHeight = newHeight  # 스크롤 높이 업데이트

bb = ""  # 빈 문자열 생성
bb = str(totalfreq[0])  # 첫 번째 빈도수를 문자열로 변환하여 bb에 저장
print("33")
print(bb)  # bb 출력
print(totalfreq[0])  # 첫 번째 빈도수 출력

# 데이터프레임 생성
df = pd.DataFrame(columns=['id', 'message', 'Date'])

number = 1  # 번호 초기화
for i in range(len(totaltweets)):
    for j in range(len(totaltweets[i])):
        # 데이터프레임에 행 추가
        df = df.append({'id': number, 'message': (totaltweets[i][j]).text, 'Date': str(totalfreq[i]).replace("'date': datetime.date", "")}, ignore_index=True)
        number = number + 1

# 엑셀 파일로 저장
writer = pd.ExcelWriter('filab_tw.xlsx', engine='xlsxwriter')

# 데이터프레임을 엑셀 파일로 변환
df.to_excel(writer, sheet_name='Sheet1')

# Pandas 엑셀 라이터 닫기 및 Excel 파일로 출력
writer.save()