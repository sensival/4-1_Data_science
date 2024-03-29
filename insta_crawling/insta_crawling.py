from selenium import webdriver 
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common import NoSuchElementException,ElementNotInteractableException
import json
import re
from bs4 import BeautifulSoup
import unicodedata
import pandas as pd

try:
    with open('C:/Users/wogns/OneDrive/바탕 화면/Data_study/데이터사이언스/insta_crawling/config.json', 'r') as file:
        config = json.load(file)

except Exception as e:
    print("Error:", e)

ID = config['DEFAULT']['LOGIN_ID']
PW = config['DEFAULT']['LOGIN_PW']



# 웹 열기
dr = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
dr.get('https://www.instagram.com/') #인스타그램 웹 켜기
time.sleep(2) 	#2초 대기

# 로그인
id_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(1) > div > label > input")   #아이디 입력창
password_box = dr.find_element(By.CSS_SELECTOR, "#loginForm > div > div:nth-child(2) > div > label > input")     #비밀번호 입력창
login_button = dr.find_element(By.CSS_SELECTOR,'#loginForm > div > div:nth-child(3) > button')      #로그인 버튼

act = ActionChains(dr)      #동작 명령어 지정
act.send_keys_to_element(id_box, ID).send_keys_to_element(password_box, PW).click(login_button).perform() #아이디 입력, 비밀 번호 입력, 로그인 버튼 클릭 수행
time.sleep(2)

#브라우저 꺼짐 방지 옵션
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

# 알림창 끄기
wait = WebDriverWait(dr, 10) # WebDriverWait 객체 생성
alert_01 = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, '_ac8f')))
alert_01.click()


alert_02 = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, '_a9_1')))
alert_02.click()
time.sleep(5)


# 해시태그
dr.get('https://www.instagram.com/explore/tags/'+ '제넥솔')
time.sleep(2)


def get_content(dr, n):
    time.sleep(1)
    list = dr.find_elements(By.CLASS_NAME,"_aagw") ##리스트를 가져와서
    list[n].click() ##리스트 요소 중 첫번째를 클릭
    time.sleep(1)
    html = dr.page_source  # 페이지 소스 가져오기
    soup = BeautifulSoup(html, 'html.parser')  # BeautifulSoup을 이용해 HTML 파싱
    content = soup.select('div._a9zs > h1')[0].text
    time.sleep(1)
    dr.back()
    
    return content

results = []
target = 3

for i in range(target):

    try:
        time.sleep(2)
        data = get_content(dr, i) #게시글 정보 가져오기
        results.append(data)

    except:
        time.sleep(2)

        

        
for i in range(target):
    print(results[i])

'''
try:
    html = dr.page_source
    soup = BeautifulSoup(html, 'html.parser')


    with open('페이지.html', 'w', encoding='utf-8') as file:

        file.write(str(soup))


except Exception as e:
    print("오류 발생:", e)
    

try:
    content = soup.select('div._a9zs > h1')[0].text
    content = unicodedata.normalize('NFC',content)

except Exception as e:
    content = ''
    print(e)





# 엑셀 파일로 저장
writer = pd.ExcelWriter('filab_tw.xlsx', engine='xlsxwriter')

# 데이터프레임을 엑셀 파일로 변환
df.to_excel(writer, sheet_name='Sheet1')

# Pandas 엑셀 라이터 닫기 및 Excel 파일로 출력 
writer.save()

lastHeight = dr.execute_script("return document.body.scrollHeight")  # 스크롤 높이 저장

while True:
        dr.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 페이지 스크롤 다운
        time.sleep(1)  # 1초 대기 페이지 로딩 기다리기

        newHeight = dr.execute_script("return document.body.scrollHeight")  # 새로운 스크롤 높이

        # 스크롤이 내려가고있으면 계속 업데이트
        if newHeight != lastHeight:
            try:
                html = dr.page_source
                soup = BeautifulSoup(html, 'html.parser')

                with open('페이지.html', 'w', encoding='utf-8') as file:
                    file.write(str(soup))


            except Exception as e:
                print("오류 발생:", e)
                

            try:
                content = soup.select('div._a9zs > h1')[0].text
                content = unicodedata.normalize('NFC',content)

            except Exception as e:
                content = ''
                print(e)

        else:
            totalfreq.append(dailyfreq)  # 날짜 추가
            wordfreq = 0  # 단어 빈도수 초기화
            startdate = untildate  # 시작 날짜 변경
            untildate += dt.timedelta(days=1)  # 다음 날짜로 변경
            dailyfreq = {}  # dailyfreq 초기화
            totaltweets.append(tweets)  # 트윗 리스트에 트윗 추가
            break

   
tags = re.findall(r'#[^\s#,\\]+', content)


date = soup.select('._a9ze._a9zf')[0]['datetime'][:10]

try:
    like = soup.select('.x193iq5w.xeuugli.x1fj9vlw.x13faqbe.x1vvkbs.xt0psk2.x1i0vuye.xvs91rp.x1s688f.x5n08af.x10wh9bi.x1wdrske.x8viiok.x18hxmgj')[0].text
    like = re.findall(r'[\d]+', like)[0] 

except:
    print('')
    print(e)


    

data = [content, date, like, tags]


def move_next(driver):

    try:

        right = driver.find_element(By.CLASS_NAME,'_aaqh')

        right.click()

        time.sleep(3)

        

        # 다음 페이지가 완전히 로드될 때까지 기다리기 위해 클릭 후 대기를 추가합니다.

        # wait = WebDriverWait(driver, 10)

        # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '다음_요소_선택자')))

        

    except Exception as e:

        print(f"오류가 발생했습니다: {e}")

​

move_next(driver)
'''